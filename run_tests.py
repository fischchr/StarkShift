#! /usr/bin/env python3

import subprocess
import sys

if '-h' in sys.argv or '--help' in sys.argv:
    print('\nRun all unittests of StarkShift module and show test coverage.')
    print('Additional commands\n')
    print('\t--html: Generate html file of code coverage.')
    print('\t--verbose: Enable logging of test results.')
    print('\t--stats: Show test coverage of last test.\n')

    raise SystemExit(0)

if '--stats' in sys.argv:
    subprocess.run(['python3', '-m', 'coverage', 'report', '-m'])
    raise SystemExit(0)


# check if we want to get coverage
#if '--coverage' in sys.argv:
#    commands += ['coverage', 'run', '-m']

# Run coverage
commands = ['python3', '-m', 'coverage', 'run', '--source=.']
# and let unittest discover all unittest files
commands += ['-m', 'unittest', 'discover']

# Add logging
if '--verbose' in sys.argv:
    commands += '-v'

subprocess.run(commands)

# Show stats
#subprocess.run(['python3', '-m', 'coverage', 'report', '-m'])

if '--html' in sys.argv:
    subprocess.run(['python3', '-m', 'coverage', 'html'])
