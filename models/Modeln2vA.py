#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import node2vec
import chainer
import chainer.functions as F
import chainer.links as L

from collections import defaultdict
import sys,random

class Module(chainer.Chain):
	def __init__(self, dim, dropout_rate, activate, isR, isBN):
		super(Module, self).__init__(
			x2z	=L.Linear(dim,dim),
			bn	=L.BatchNormalization(dim)
		)
		self.dropout_rate=dropout_rate
		self.activate = activate
		self.is_residual = isR
		self.is_batchnorm = isBN
	def __call__(self, x):
		if self.dropout_rate!=0:
			x = F.dropout(x,ratio=self.dropout_rate)
		z = self.x2z(x)
		if self.is_batchnorm:
			z = self.bn(z)
		if self.activate=='tanh': z = F.tanh(z)
		if self.activate=='relu': z = F.relu(z)
		if self.is_residual:	return z+x
		else: return z
class Block(chainer.Chain):
	def __init__(self, dim, dropout_rate, activate, layer, isR, isBN):
		super(Block, self).__init__()
		links = [('m{}'.format(i), Module(dim,dropout_rate,activate, isR, isBN)) for i in range(layer)]
		for link in links:
			self.add_link(*link)
		self.forward = links
	def __call__(self,x):
		for name, _ in self.forward:
			x = getattr(self,name)(x)
		return x

class Tunnel(chainer.Chain):
	def __init__(self, dim, dropout_rate, activate, layer, isR, isBN, relation_size, pooling_method):
		super(Tunnel, self).__init__()
		linksH = [('h{}'.format(i), Block(dim,dropout_rate,activate,layer,isR,isBN)) for i in range(relation_size)]
		for link in linksH:
			self.add_link(*link)
		self.forwardH = linksH
		linksT = [('t{}'.format(i), Block(dim,dropout_rate,activate,layer,isR,isBN)) for i in range(relation_size)]
		for link in linksT:
			self.add_link(*link)
		self.forwardT = linksT
		self.pooling_method = pooling_method
		self.layer = layer

	def maxpooling(self,xs,neighbor):
		sources = defaultdict(list)
		for ee in neighbor:
			for i in neighbor[ee]:
				sources[i].append(xs[ee])
		result = []
		for i,xxs in sorted(sources.items(),key=lambda x:x[0]):
			if len(xxs)==1: result.append(xxs[0])
			else:
				x = F.concat(xxs,axis=0)					# -> (b,d)
				x = F.swapaxes(x,0,1)						# -> (d,b)
				x = F.maxout(x,len(xxs))					# -> (d,1)
				x = F.swapaxes(x,0,1)						# -> (1,d)
				result.append(x)
		return result

	def averagepooling(self,xs,neighbor):
		sources = defaultdict(list)
		for ee in neighbor:
			for i in neighbor[ee]:
				sources[i].append(xs[ee])
		result = []
		for i,xxs in sorted(sources.items(),key=lambda x:x[0]):
			if len(xxs)==1: result.append(xxs[0])
			else:			result.append(sum(xxs)/len(xxs))
		return result

	def sumpooling(self,xs,neighbor):
		sources = defaultdict(list)
		for ee in neighbor:
			for i in neighbor[ee]:
				sources[i].append(xs[ee])
		result = []
		for i,xxs in sorted(sources.items(),key=lambda x:x[0]):
			if len(xxs)==1: result.append(xxs[0])
			else:			result.append(sum(xxs))
		return result

	def easy_case(self,x,neighbor_entities,neighbor_dict,assign,entities,relations):
		x = F.split_axis(x,len(neighbor_dict),axis=0)

		assignR = dict()
		bundle = defaultdict(list)
		for v,k in enumerate(neighbor_entities):
			for i in assign[v]:
				e = entities[i]
				if (e,k) in relations:	r =  relations[(e,k)]*2
				else:					r =  relations[(k,e)]*2+1
				assignR[(r,len(bundle[r]))] = v
				bundle[r].append(x[v])

		result = [0 for i in range(len(neighbor_dict))]
		for r in bundle:
			rx = bundle[r]
			if len(rx)==1:	result[assignR[(r,0)]] = rx[0]
			else:
				for i,x in enumerate(rx):
					result[assignR[(r,i)]] = x

		if self.pooling_method=='max':
			result = self.maxpooling(result,assign)
		if self.pooling_method=='avg':
			result = self.averagepooling(result,assign)
		if self.pooling_method=='sum':
			result = self.sumpooling(result,assign)
		result = F.concat(result,axis=0)
		return result



	"""
	# neighbor_entities=[(k,v)]
	# (e,k) in links
	# e = entities[i]
	# i in assing[v]
	"""
	"""
	source entityから出てるedgeが無い
	"""
	def __call__(self,x,neighbor_entities,neighbor_dict,assign,entities,relations):
		if self.layer==0:
			return self.easy_case(x,neighbor_entities,neighbor_dict,assign,entities,relations)

		if len(neighbor_dict)==1:
			x=[x]
		else:
			x = F.split_axis(x,len(neighbor_dict),axis=0)

		assignR = dict()
		bundle = defaultdict(list)
		for v,k in enumerate(neighbor_entities):
			for i in assign[v]:
				e = entities[i]
				if (e,k) in relations:	r =  relations[(e,k)]*2
				else:					r =  relations[(k,e)]*2+1
				assignR[(r,len(bundle[r]))] = v
				bundle[r].append(x[v])

		result = [0 for i in range(len(neighbor_dict))]
		for r in bundle:
			rx = bundle[r]
			if len(rx)==1:
				rx=rx[0]
				if r%2==0:	rx = getattr(self,self.forwardH[r//2][0])(rx)
				else:		rx = getattr(self,self.forwardT[r//2][0])(rx)
				result[assignR[(r,0)]] = rx
			else:
				size = len(rx)
				rx = F.concat(rx,axis=0)
				if r%2==0:	rx = getattr(self,self.forwardH[r//2][0])(rx)
				else:		rx = getattr(self,self.forwardT[r//2][0])(rx)
				rx = F.split_axis(rx,size,axis=0)
				for i,x in enumerate(rx):
					result[assignR[(r,i)]] = x

		if self.pooling_method=='max':
			result = self.maxpooling(result,assign)
		if self.pooling_method=='avg':
			result = self.averagepooling(result,assign)
		if self.pooling_method=='sum':
			result = self.sumpooling(result,assign)
		result = F.concat(result,axis=0)
		return result

class AModule(chainer.Chain):
	def __init__(self, dim):
		super(AModule, self).__init__(
			es2e=L.Linear(dim*2,1)
		)
		self.dim=dim
	def __call__(self,e12):
		eij=self.es2e(e12)
		return eij
class Attention(chainer.Chain):
	def __init__(self, heads_size,dim):
		super(Attention,self).__init__()
		linksA=[('a{}'.format(i),AModule(dim)) for i in range(heads_size)]
		for link in linksA:
			self.add_link(*link)
		self.dim=dim
		self.forwardA=linksA
		self.heads_size=heads_size

	def __call__(self,result,n2v_neighbors):
		if len(neigbor_dict)==1:
			x=[x]
		else:
			x=F.split_axis(x,len(neighbor_dict),axis=0)

		es=defaultdict(list)
		for v in range(len(x)):
			for j in range(self.heads_size):
				name=self.forwardA[j][0]
				for k in assign[v]:
					eij=getattr(self,name)(x[v],assign[v])
					es[j].append(eij)
				for k in range(len(assign[v])-1):
					es[j][0]=F.concat((es[j][0],es[j][k+1]),axis=0)
				e[j][0]=e[j][0].reshape(1,e[j][0].shape[0])
				e[j][0]=F.softmax(e[j][0])
			for j in range(self.heads_size-1):
				e[0][0]=e[0][0]+e[j+1][0]
			e[0][0]=e[0][0]/self.heads_size
			x[v]=self.attentionpooling(v,x,assign,e[0][0],neighbor_dict)
		x=F.concat(x,axis=0)
		return x

class N2VModule(chainer.Chain):
	def __init__(self,dim,walk_length):
		super(N2VModule,self).__init__(
				es2e=L.Linear(dim*walk_length,dim)
			)
	def __call__(self,es):
		x=self.es2e(es)
		y=F.relu(x)
		return y

class N2V(chainer.Chain):
	def __init__(self,N2V_size,dim,walk_length):
		super(N2V,self).__init__()
		linksN=[('a{}'.format(i),N2VModule(dim,walk_length)) for i in range(N2V_size)]
		for link in linksN:
			self.add_link(*link)
		self.forwardN=linksN
		self.N2V_size=N2V_size
		self.walk_length=walk_length

	def __call__(self,neighbors):

		result=defaultdict()
		for h in range(self.N2V_size):
			name=self.forwardN[h][0]
			rx=getattr(self,name)(neighbors)
			if h==0:	result=rx
			else:		result=result+rx
		result=result/self.N2V_size
		return result


class Model(chainer.Chain):
	def __init__(self, args):
		super(Model, self).__init__(
			embedE	= L.EmbedID(args.entity_size,args.dim),
			embedR	= L.EmbedID(args.rel_size,args.dim),
		)
		linksB = [('b{}'.format(i), Tunnel(args.dim,args.dropout_block,args.activate,args.layerR,args.is_residual,args.is_batchnorm, args.rel_size, args.pooling_method)) for i in range(args.order)]
		for link in linksB:
			self.add_link(*link)

		self.forwardNN=N2V(args.N2V_size,args.dim,args.walk_length)



		self.sample_size = args.sample_size
		self.dropout_embed = args.dropout_embed
		self.dropout_decay = args.dropout_decay
		self.depth = args.order
		self.is_embed = args.is_embed
		self.is_known = args.is_known
		self.threshold = args.threshold
		self.objective_function = args.objective_function
		self.is_bound_wr = args.is_bound_wr

		self.heads_size=args.heads_size
		self.dim=args.dim
		self.forwardAA=Attention(args.heads_size,args.dim)

		self.walk_length=args.walk_length


		if args.use_gpu: self.to_gpu()


	def get_context(self,entities,links,relations,edges,order,xp,n2v_neighbors):
		result=[]
		neighbors=[0 for i in range(len(entities))]
		for i in range(len(entities)):
			neighbor=self.embedE(xp.array(n2v_neighbors[entities[i]],'i'))
			neighbors[i]=neighbor.reshape(1,self.walk_length*self.dim)

		neighbors=F.concat(neighbors,axis=0)
		
		result=self.forwardNN(neighbors)

		return result

	def train(self,positive,negative,links,relations,edges,xp,n2v_neighbors):
		self.cleargrads()

		entities= set()
		for h,r,t in positive:
			entities.add(h)
			entities.add(t)
		for h,r,t in negative:
			entities.add(h)
			entities.add(t)

		entities = list(entities)

		x = self.get_context(entities,links,relations,edges,0,xp,n2v_neighbors)
		x = F.split_axis(x,len(entities),axis=0)
		edict = dict()
		for e,x in zip(entities,x):
			edict[e]=x

		pos,rels = [],[]
		for h,r,t in positive:
			rels.append(r)
			pos.append(edict[h]-edict[t])
		pos = F.concat(pos,axis=0)
		xr = self.embedR(xp.array(rels,'i'))
		if self.is_bound_wr:	xr = F.tanh(xr)
		pos = F.batch_l2_norm_squared(pos+xr)

		neg,rels = [],[]
		for h,r,t in negative:
			rels.append(r)
			neg.append(edict[h]-edict[t])
		neg = F.concat(neg,axis=0)
		xr = self.embedR(xp.array(rels,'i'))
		if self.is_bound_wr:	xr = F.tanh(xr)
		neg = F.batch_l2_norm_squared(neg+xr)

		if self.objective_function=='relative': return sum(F.relu(self.threshold+pos-neg))
		if self.objective_function=='absolute': return sum(pos+F.relu(self.threshold-neg))


	def get_scores(self,candidates,links,relations,edges,xp,mode,n2v_neighbors):
		entities = set()
		for h,r,t,l in candidates:
			entities.add(h)
			entities.add(t)
		entities = list(entities)
		xe = self.get_context(entities,links,relations,edges,0,xp,n2v_neighbors)
		xe = F.split_axis(xe,len(entities),axis=0)
		edict = dict()
		for e,x in zip(entities,xe):
			edict[e]=x
		diffs,rels = [],[]
		for h,r,t,l in candidates:
			rels.append(r)
			diffs.append(edict[h]-edict[t])
		diffs = F.concat(diffs,axis=0)
		xr = self.embedR(xp.array(rels,'i'))
		if self.is_bound_wr:	xr = F.tanh(xr)
		scores = F.batch_l2_norm_squared(diffs+xr)
		return scores
