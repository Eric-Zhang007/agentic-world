"""Compare untouched remote source worlds with their pre-trial local copies."""
import json
from pathlib import Path
import shlex
import sqlite3
from contextlib import closing

from run_gta_pi_cases import LOCAL, OUT, connect


def snapshot(path):
    with closing(sqlite3.connect(Path(path).as_uri()+'?mode=ro',uri=True)) as db:
        return {table:[list(row) for row in db.execute(f'SELECT * FROM {table} ORDER BY 1')]
                for table in ['objects','world_meta','evidence','edits']}


def main():
    names=['rural/continuation','urban']
    code='import json,sqlite3,pathlib\nresult={}\n'
    code+=f'for name in {names!r}:\n'
    code+=f' db=sqlite3.connect((pathlib.Path({OUT!r})/name/"world.sqlite").as_uri()+"?mode=ro",uri=True)\n'
    code+=' result[name]={table:db.execute("SELECT * FROM "+table+" ORDER BY 1").fetchall() for table in ["objects","world_meta","evidence","edits"]}\n db.close()\n'
    code+='print(json.dumps(result))\n'
    client=connect()
    try:
        _,stdout,stderr=client.exec_command('python3 -c '+shlex.quote(code))
        raw=stdout.read().decode()
        if stdout.channel.recv_exit_status():raise RuntimeError(stderr.read().decode())
    finally:client.close()
    remote=json.loads(raw)
    result={}
    for name in names:
        before=snapshot(LOCAL/name/'world.sqlite')
        result[name]={table:before[table]==remote[name][table] for table in before}
    (LOCAL/'source_world_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not all(all(tables.values()) for tables in result.values()):raise SystemExit(1)


if __name__=='__main__':main()
