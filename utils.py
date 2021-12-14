import argparse
import json

class listToVal(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        setattr(args, self.dest, values if len(values) > 1 else values[0])
 
def json2obj(mod, cls, **pars):
	return getattr(mod, cls)(**pars)

def file2obj(mod, file_name, **extra_args):

	with open(file_name, 'rt') as f:
		pars = json.load(f)

	return json2obj(mod, pars['cls'], **pars['args'], **extra_args)