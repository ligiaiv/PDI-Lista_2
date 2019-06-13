outp = open("emolexDictionaryPortNew.txt",'w')
with open("emolexDictionaryPort.txt",'r') as inp:
	for line in inp:
		outp.write(line.lower())
outp.close()
