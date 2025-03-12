include configs/parczech2tsv.mk

parczechAnaFilename := ParCzech$(parczechVersion).TEI.ana.tar.gz
parczechAnaURL:=https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/$(parczechHandleID)/$(parczechAnaFilename)
parczechAnaFilePath := $(data)/$(parczechAnaFilename)
parczechAnaDirPath := $(data)/ParCzech.TEI.ana
parczechAnaRootPath := $(parczechAnaDirPath)/ParCzech.ana.xml

### getting parczech data from lindat repository
$(parczechAnaFilePath):
	mkdir -p $(data)
	wget -O $(parczechAnaFilePath) $(parczechAnaURL)
parczech-download: | $(parczechAnaFilePath)


$(parczechAnaDirPath): $(parczechAnaFilePath)
	tar -xvzf $(parczechAnaFilePath) -C $(data)
parczech-unpack: | $(parczechAnaDirPath)

prepare-parczech-data: parczech-unpack

### converting to TSV





