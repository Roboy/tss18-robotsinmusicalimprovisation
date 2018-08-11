#!/bin/bash
#Dependencies:
#MuseScore 2.2

for file in ~/Uni/Wikifonia/*.mxl; do
	#echo $filename
	filename=$(basename -- "$file")
	#extension="${filename##*.}"
	filenameNoExtension="${filename%.*}"

	echo $filename
	#echo $filenameNoExtension.$extension
	../../../../Applications/MuseScore\ 2.app/Contents/MacOS/mscore $filename -o ~/Uni/WikifoniaMidi/$filenameNoExtension.mid
done

#END