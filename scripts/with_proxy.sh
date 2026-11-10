#!/bin/sh
# Scope proxy routing to this process and its children.
export HTTP_PROXY=http://127.0.0.1:17890
export HTTPS_PROXY=http://127.0.0.1:17890
export NO_PROXY=localhost,127.0.0.1,::1
exec "$@"
