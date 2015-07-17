import subprocess

__all__ = [ "sbatch" ]

def sbatch( cmd, mem=5, c=10, verbose=False, dryrun=False, partition=None ):
    sbatch_cmd = 'sbatch '
    if partition is not None:
        sbatch_cmd += '-p ' + partition + ' '
    sbatch_cmd += '--mem='+str(mem)+'G -c '+str(c)+' --wrap="'+' '.join(cmd)+'"'

    if verbose or dryrun:
        print sbatch_cmd

    if dryrun:
        return
    
    return subprocess.call( sbatch_cmd, shell=True )
    
