"""Coarse walking support and body clearance against recorded triangles."""
from functools import lru_cache
from pathlib import Path
import math
import numpy as np


class SurfaceWorld:
    def __init__(self, triangles):
        self.triangles = np.asarray(triangles, dtype=float).reshape(-1,3,3)
        self.lo = self.triangles.min(axis=1)
        self.hi = self.triangles.max(axis=1)
        normals = np.cross(self.triangles[:,1]-self.triangles[:,0],self.triangles[:,2]-self.triangles[:,0])
        lengths = np.linalg.norm(normals,axis=1)
        self.walkable = np.abs(normals[:,1]) / np.maximum(lengths,1e-12) >= math.cos(math.radians(50))
        self.cells = {}
        self.large = []
        for i, (lo,hi) in enumerate(zip(self.lo,self.hi)):
            a,b = np.floor(lo[[0,2]]/2).astype(int),np.floor(hi[[0,2]]/2).astype(int)
            if np.prod(b-a+1)>256:
                self.large.append(i)
                continue
            for x in range(a[0],b[0]+1):
                for z in range(a[1],b[1]+1):self.cells.setdefault((x,z),[]).append(i)

    def candidates(self,x,z,radius=0):
        ids=set(self.large)
        for ix in range(math.floor((x-radius)/2),math.floor((x+radius)/2)+1):
            for iz in range(math.floor((z-radius)/2),math.floor((z+radius)/2)+1):ids.update(self.cells.get((ix,iz),()))
        return np.asarray(sorted(ids),dtype=int)

    def floor(self,x,z,reference,max_step=.35,max_drop=.5):
        ids=self.candidates(x,z)
        ids=ids[self.walkable[ids]]
        t=self.triangles[ids]
        if not len(t):return None
        a=t[:,0];u=t[:,1]-a;v=t[:,2]-a
        dx=x-a[:,0];dz=z-a[:,2]
        det=u[:,0]*v[:,2]-u[:,2]*v[:,0]
        good=np.abs(det)>1e-10
        s=np.divide(dx*v[:,2]-dz*v[:,0],det,out=np.zeros_like(det),where=good)
        q=np.divide(u[:,0]*dz-u[:,2]*dx,det,out=np.zeros_like(det),where=good)
        y=a[:,1]+s*u[:,1]+q*v[:,1]
        good &= (s>=-1e-7)&(q>=-1e-7)&(s+q<=1+1e-7)&(y<=reference+max_step+1e-7)&(y>=reference-max_drop-1e-7)
        return float(y[good].max()) if good.any() else None

    def body_blocked(self,position,size,foot_clearance=.35):
        # Triangle/AABB separating-axis test; body is deliberately coarse.
        p=np.asarray(position,float);size=np.asarray(size,float)
        radius=max(size[0],size[2])/2
        rounded_foot=foot_clearance<.01
        if rounded_foot:
            ids=self.candidates(p[0],p[2],radius)
            center=p+[0,radius,0]
            ids=ids[np.all(self.hi[ids]>=center-radius,axis=1)&np.all(self.lo[ids]<=center+radius,axis=1)]
            t=self.triangles[ids]
            if len(t):
                a=t[:,0];u=t[:,1]-a;v=t[:,2]-a;w=center-a
                uu=np.einsum('ij,ij->i',u,u);vv=np.einsum('ij,ij->i',v,v);uv=np.einsum('ij,ij->i',u,v)
                wu=(w*u).sum(axis=1);wv=(w*v).sum(axis=1)
                den=uu*vv-uv*uv;valid=np.abs(den)>1e-12
                s=np.divide(wu*vv-wv*uv,den,out=np.zeros_like(den),where=valid)
                q=np.divide(wv*uu-wu*uv,den,out=np.zeros_like(den),where=valid)
                inside=valid&(s>=0)&(q>=0)&(s+q<=1)
                residual=w-s[:,None]*u-q[:,None]*v
                distance=np.where(inside,np.einsum('ij,ij->i',residual,residual),np.inf)
                for j in range(3):
                    edge=t[:,(j+1)%3]-t[:,j];offset=center-t[:,j]
                    factor=np.clip(np.einsum('ij,ij->i',offset,edge)/np.maximum(np.einsum('ij,ij->i',edge,edge),1e-12),0,1)
                    residual=offset-factor[:,None]*edge
                    distance=np.minimum(distance,np.einsum('ij,ij->i',residual,residual))
                if np.any(distance<(radius-foot_clearance)**2):return True
            foot_clearance=radius
        foot_clearance=min(foot_clearance,size[1]*.25)
        half=np.array([radius,(size[1]-foot_clearance)/2,radius])
        center=p+[0,foot_clearance+half[1],0]
        ids=self.candidates(p[0],p[2],max(half[0],half[2]))
        ids=ids[np.all(self.hi[ids]>=center-half,axis=1)&np.all(self.lo[ids]<=center+half,axis=1)]
        t=self.triangles[ids]-center
        if not len(t):return False
        edges=np.roll(t,-1,axis=1)-t
        axes=[np.broadcast_to(axis,(len(t),3)) for axis in np.eye(3)]
        axes.append(np.cross(edges[:,0],edges[:,1]))
        axes.extend(np.cross(edges[:,i],axis) for i in range(3) for axis in np.eye(3))
        overlap=np.ones(len(t),dtype=bool)
        for axis in axes:
            projection=np.einsum('nij,nj->ni',t,axis)
            extent=np.abs(axis)@half
            overlap &= (projection.min(axis=1)<=extent+1e-8)&(projection.max(axis=1)>=-extent-1e-8)
        return bool(overlap.any())


@lru_cache(maxsize=4)
def _load(path,mtime):
    return SurfaceWorld(np.fromfile(path,dtype='<f4').reshape(-1,3,3))


def surface_world(scene):
    spec=scene.get('static_geometry',{})
    path=spec.get('collision_path')
    if not path:
        if spec:raise ValueError('Imported scenery needs collision_path before motion simulation')
        return None
    path=Path(path).resolve()
    return _load(str(path),path.stat().st_mtime_ns)


def floor_height(scene,x,z,reference):
    world=surface_world(scene)
    if world:return world.floor(x,z,reference)
    from whitebox.world import terrain_height
    return terrain_height(scene['terrain'],x,z)
