#! -*- coding:utf-8 -*-

from models.ModelA0 import Model as A0
from models.Modelm0 import Model as m0
from models.Modelm2 import Model as m2
from models.Modeln2v import Model as n2v
from models.Modeln2vA import Model as n2vA

import sys
def get_model(args):
	if args.nn_model=='A0':			return A0(args)
	if args.nn_model=='m0':			return m0(args)
	if args.nn_model=='m2':			return m2(args)
	if args.nn_model=='n2v':		return n2v(args)	
	if args.nn_model=='n2vA':		return n2vA(args)
	print('no such model:',args.nn_model)
	sys.exit(1)
