import os

def import_idt(file):
	tar_len = 15;

	f = open('test.bin', "rb");

	# feat = f.read(1);
	# while True:

	# a = f.readline();
	print os.stat('test.bin')
	# print f.readline();
	# print type()
	# print f.readline();
	# print ord(feat)
	f.close()


import_idt('test.bin');