include configs/parczech2tsv.mk

parczechAnaFilename := ParCzech$(parczechVersion).TEI.ana.tar.gz
parczechAnaURL:=https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/$(parczechHandleID)/$(parczechAnaFilename)
parczechAnaFilePath := $(data)/$(parczechAnaFilename)
parczechAnaDirPath := $(data)/ParCzech.TEI.ana
parczechAnaRootPath := $(parczechAnaDirPath)/ParCzech.ana.xml

saxon := ./scripts/bin/saxon.jar

### check and install prerequisites

check-prereq:
	@uname -a|grep -iq ubuntu || \
	  ( echo -n "WARN: not running on ubuntu-derived system: " && uname -a )
	@echo -n "Saxon: "
	@test -f $(saxon) && \
	  unzip -p $(saxon) META-INF/MANIFEST.MF|grep 'Main-Class:'| grep -q 'net.sf.saxon.Transform' && \
	  echo "OK" || echo "FAIL"

prereq-setup-saxon:
	@mkdir -p ./scripts/bin/
	@wget -O saxon.zip https://github.com/Saxonica/Saxon-HE/releases/download/SaxonHE12-5/SaxonHE12-5J.zip
	@unzip -p saxon.zip saxon-he-12.5.jar > ./scripts/bin/saxon.jar
	@rm saxon.zip
	@echo "Saxon installed in $(saxon)"

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





