import gensim
import os
import codecs
from sklearn.manifold import TSNE
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
import sys
from sklearn.neighbors import NearestNeighbors
from util import check_if_token_is_method_signature
from util import check_if_token_is_object_signature


def check_package_include(packages,text):
	
	check = False
	for package in packages:
		if package in text:
			check = True
	return check

project = "sdk"
cs_packages = ["system.","nunit.","microsoft."]
java_packages = ["java.","javax.", "sun.", "w3c.","jdk.", "junit."]


is_exact_name = False

top_k = 5
usage_type = "method"

if usage_type == "method":
	func = check_if_token_is_method_signature
else:
	func = check_if_token_is_object_signature
# cs_embeddings_path = "./embeddings/cs_biskip.txt"
# java_embeddings_path = "./embeddings/java_biskip.txt"

cs_embeddings_path = "./embeddings/vectors-cs.txt"
java_embeddings_path = "./embeddings/vectors-java.txt"

with open(cs_embeddings_path) as cs_f:
	# next(cs_f)
	cs_embeddings = cs_f.readlines()

with open(java_embeddings_path) as java_f:
	# next(java_f)
	java_embeddings = java_f.readlines()

cs_signature_tokens = list()
java_signature_tokens = list()

for cs_emb in cs_embeddings:
	split = cs_emb.split(" ")
	if func(split[0]) == True:
		if check_package_include(cs_packages,split[0]) == True:
			# if split[0][0] == "S":
		# if "System." in split[0] or "antlr" in split[0].lower():
			cs_signature_tokens.append(split[0])

print "cs tokens : " + str(len(cs_signature_tokens))

for java_emb in java_embeddings:
	split = java_emb.split(" ")
	if func(split[0]) == True:

		if check_package_include(java_packages,split[0]) == True:
		# if "java." in split[0] or "antlr" in split[0].lower():
			java_signature_tokens.append(split[0])

print "java tokens : " + str(len(java_signature_tokens))
print "Loading word embedding..........."
# cs_vectors = KeyedVectors.load_word2vec_format("./bi2vec_vectors/cs_vectors_new.txt",binary=False)
# java_vectors = KeyedVectors.load_word2vec_format("./bi2vec_vectors/java_vectors_new.txt",binary=False)

cs_vectors = KeyedVectors.load_word2vec_format(cs_embeddings_path,binary=False)
java_vectors = KeyedVectors.load_word2vec_format(java_embeddings_path,binary=False)

print "Finish loading.............."


for cs_token in cs_signature_tokens:
	split = cs_token.split(".")
	method_source = split[len(split)-1].split("(")[0]

	k_nearest = java_vectors.similar_by_vector(cs_vectors[cs_token], topn=top_k)
	relevant_k = list()
	for k in k_nearest:
		if func(k[0]) == True:

			# if check_package_include(java_packages,k[0]) == True:

			if is_exact_name:
				split = k[0].split(".")
				method_target = split[len(split) - 1].split("(")[0]
				print "comparing : " + method_source + " vs " + method_target
				if method_target.lower() == method_source.lower(): 
			
					relevant_k.append(k[0])

			else:
				relevant_k.append(k[0])

	if is_exact_name:
		outpur_cs_url = "./usage_mapping/" + project + "_" + usage_type + "_usage_mapping_cs_exact_name_" + str(top_k) + ".txt"
	else:
		outpur_cs_url = "./usage_mapping/" + project + "_" + usage_type + "_usage_mapping_cs_" + str(top_k) + ".txt"
	if len(relevant_k) != 0:
		with open(outpur_cs_url,"a") as f1:
			f1.write(cs_token + "-" + "**".join(relevant_k) + "\n")
			f1.write(cs_token + "-" + relevant_k[0] + "\n")
			f1.write(cs_token + "-" + relevant_k[0] + "\n")

for java_token in java_signature_tokens:
	split = java_token.split(".")
	method_source = split[len(split)-1].split("(")[0]
	k_nearest = cs_vectors.similar_by_vector(java_vectors[java_token], topn=top_k)
	relevant_k = list()
	for k in k_nearest:
		if func(k[0]) == True:
			# if check_package_include(cs_packages,k[0]) == True:

			if is_exact_name:
				split = k[0].split(".")
				method_target = split[len(split) - 1].split("(")[0]
			# if "java." in k[0] or "antlr" in k[0].lower():
				print "comparing : " + method_source + " vs " + method_target
				if method_target.lower() == method_source.lower(): 
					relevant_k.append(k[0])
			else:
				relevant_k.append(k[0])


	if is_exact_name:
		outpur_java_url = "./usage_mapping/" + project + "_" + usage_type + "_usage_mapping_java_exact_" + str(top_k) + ".txt"
	else:
		outpur_java_url = "./usage_mapping/" + project + "_" + usage_type + "_usage_mapping_java_" + str(top_k) + ".txt"

	if len(relevant_k) != 0:
		with open(outpur_java_url,"a") as f1:
			try:
				f1.write(java_token + "-" + "**".join(relevant_k) + "\n")
			except Exception as e:
				print(e)
			# f1.write(java_token + "-" + relevant_k[0] + "\n")
			# f1.write(java_token + "-" + relevant_k[0] + "\n")