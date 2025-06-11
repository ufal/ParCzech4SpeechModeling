<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:fo="http://www.w3.org/1999/XSL/Format"
  xmlns="http://www.tei-c.org/ns/1.0"
  xmlns:tei="http://www.tei-c.org/ns/1.0"
  exclude-result-prefixes="tei">
    <xsl:output method="text" omit-xml-declaration="yes" indent="no"/>
    <xsl:key name="whenID" match="tei:timeline/tei:when" use="@xml:id"/>
    <xsl:template match="tei:pb">
        <xsl:text># AUDIO: </xsl:text>
        <xsl:variable name="audio-id" select="substring-after(@corresp,'#')" />
        <xsl:value-of select="//tei:media[@xml:id=$audio-id]/@url" />
        <xsl:text>&#xA;</xsl:text>
    </xsl:template>
      <xsl:variable name="date" select="/tei:TEI//tei:settingDesc/tei:setting/tei:date[contains(concat(' ',./@ana,' '),' #parla.sitting ')]/@when"/>
    <xsl:template match="*[local-name(.) = 's']">
        <xsl:apply-templates />
        <xsl:text>&#xA;</xsl:text>
    </xsl:template>
    <xsl:template match="*[local-name(.) = 'w' or local-name(.) = 'pc' ]">
        <xsl:variable name="wid" select="@xml:id"/>
        <xsl:value-of select="concat(text(),'&#x9;',$wid)"/>
        <xsl:value-of select="concat('&#x9;',substring-after(ancestor::tei:u/@who,'#'))"/><!-- speaker ID -->
        <xsl:value-of select="concat('&#x9;',ancestor::tei:u/@*[local-name(.) = 'id'])"/><!-- speech ID -->
        <xsl:value-of select="concat('&#x9;',$date)"/><!-- date -->
        <xsl:value-of select="string('&#x9;')"/>
        <xsl:choose><!-- no space after -->
            <xsl:when test="@join">True</xsl:when>
            <xsl:otherwise>False</xsl:otherwise>
        </xsl:choose>
        <xsl:value-of select="string('&#x9;')"/>
        <xsl:choose><!-- is word -->
            <xsl:when test="local-name(.) = 'w'">True</xsl:when>
            <xsl:otherwise>False</xsl:otherwise>
        </xsl:choose>
        <xsl:variable name="startSynch" select ="preceding-sibling::tei:anchor[1][starts-with(@synch,concat('#',$wid,'.'))]/@synch" />
		<xsl:variable name="endSynch" select="following-sibling::tei:anchor[1][starts-with(@synch,concat('#',$wid,'.'))]/@synch" />
        <xsl:value-of select="string('&#x9;')"/>
        <xsl:choose><!-- start synch -->
            <xsl:when test="$startSynch"><xsl:value-of select="key('whenID', substring($startSynch, 2))/@interval" /></xsl:when>
            <xsl:otherwise>-</xsl:otherwise>
        </xsl:choose>
        <xsl:value-of select="string('&#x9;')"/>
        <xsl:choose><!-- end synch -->
            <xsl:when test="$endSynch"><xsl:value-of select="key('whenID', substring($endSynch, 2))/@interval" /></xsl:when>
            <xsl:otherwise>-</xsl:otherwise>
        </xsl:choose>
        <xsl:value-of select="string('&#xA;')"/>
    </xsl:template>
    <xsl:template match="text()" />
</xsl:stylesheet>