import argparse
import os

class PathExists(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not os.path.exists( values ):
            raise ValueError("path " + values + " does not exists")
        setattr(namespace, self.dest, values)

