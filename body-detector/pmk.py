#!/usr/bin/env python

import paramiko as pmk
import select
import os
import argparse
import socket
import time

from joblib import Parallel, delayed

parser = argparse.ArgumentParser(
description="""Paramiko script to start ipengines on lab machines through ssh.
The ipcontroller needs to be started on the master machine before starting the
ipengines on the slave machines:
    ipcontroller --ip=* --location=biomedia11.doc.ic.ac.uk
For more information on IPython.parallel, visit:
    http://ipython.org/ipython-doc/stable/parallel/
""", formatter_class=argparse.RawDescriptionHelpFormatter )
parser.add_argument( "--start", action="store_true", default=False,
                     help='start ipengines')
parser.add_argument( "--location", type=str, default="biomedia11.doc.ic.ac.uk",
                     help='host for ipcontroller')
parser.add_argument( "--threshold", type=float, default=0.80,
                     help='only use machines which have more than threshold memory free')
parser.add_argument( "--free", action="store_true", default=False,
                     help='list free memory on lab machines')
parser.add_argument( "--kill", action="store_true", default=False,
                     help='kill all running processes')
parser.add_argument( "--kill_python", action="store_true", default=False,
                     help='kill all python processes')
parser.add_argument( "--n_jobs", type=int, default=20,
                     help='maximum number of threads')
parser.add_argument( '--debug', action="store_true", default=False )
args = parser.parse_args()

hostnames = ( ["line{0:02d}.doc.ic.ac.uk".format(x+1) for x in range(33)] +
              ["ray{0:02d}.doc.ic.ac.uk".format(x+1) for x in range(40)] +
              ["corona{0:02d}.doc.ic.ac.uk".format(x+1) for x in range(40)] )

# This will report the percentage of memory that's free
# http://stackoverflow.com/questions/10585978/linux-command-for-percentage-of-memory-that-is-free
free_cmd = "free | grep Mem | awk '{print $4/$2 * 100.0}'"

def runssh(cmd, host):
    sshErrorFlag = True
    ssh = pmk.SSHClient()
    ssh.set_missing_host_key_policy(pmk.AutoAddPolicy())
    ssh.load_system_host_keys()

    try:
        ssh.connect( host,
                     username='kpk09',
                     timeout=4 )
    except ( pmk.BadHostKeyException,
             pmk.AuthenticationException,
             pmk.SSHException,
             socket.error ) as error_message:
            print error_message, host
            return
            
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(cmd)

    return ssh_stdout.readlines()

def runssh2(cmd, host, folder=None):
    sshErrorFlag = True
    ssh = pmk.SSHClient()
    ssh.set_missing_host_key_policy(pmk.AutoAddPolicy())
    ssh.load_system_host_keys()

    try:
        ssh.connect( host,
                     username='kpk09',
                     timeout=4 )
    except ( pmk.BadHostKeyException,
             pmk.AuthenticationException,
             pmk.SSHException,
             socket.error ) as error_message:
            print error_message, host
            return
    
    channel = ssh.invoke_shell()
    # channel.send('/bin/bash\n')
    # channel.send('source ~/.bash_setup\n')
    # if folder is not None:
    #     channel.send('cd '+ folder + '\n')
    channel.send(cmd.rstrip()+'\n')
    output=channel.recv(2024)
    print(output)
    time.sleep(10)
    return

if args.free:
    for host in hostnames:
        output = runssh(free_cmd,host)
        if output is not None:
            print host, float(output[0].rstrip())

if args.start:
    print "Did you remember to start the ipcontroller?"
    print "ipcontroller --ip=* --location="+ args.location
    def start(host):
        cmd = 'echo "source ~/.bash_setup && ipengine --file=.ipython/profile_default/security/ipcontroller-engine.json --location='+args.location+' &" | /bin/bash -s'
        output = runssh(free_cmd,host)
        if output is not None and float(output[0].rstrip()) > args.threshold:
            output = runssh2(cmd,host)
            if output is not None:
                print host, output

    Parallel(n_jobs=args.n_jobs)(delayed(start(host) for host in hostnames )

if args.kill:
    for host in hostnames:
        print "killing", host
        os.system('ssh -o StrictHostKeyChecking=no -o ConnectTimeout=2 {0} "pkill -u kpk09"'.format(host))

if args.kill_python:
    for host in hostnames:
        print "killall python", host
        os.system('ssh -o StrictHostKeyChecking=no -o ConnectTimeout=2 {0} "killall python"'.format(host))
