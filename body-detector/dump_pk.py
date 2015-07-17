#!/usr/bin/python

import sys
import cPickle as pickle
import pprint

x = pickle.load( open(sys.argv[1],'rb') )
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(x)
