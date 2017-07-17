#!/bin/bash

year=2017
folder=''
subfolder=''
name=''
title=''
keyword=''

meta=1

while read line
do
    # meta: ignore until first "##"
    if [[ meta = 1 && "${line:0:2}" != "##" ]]
    then
	continue
    fi
    meta=0

    if [ -n "$line" ]
    then
        # header
        if [ "${line:0:3}" == "## " ]
        then
	   folder="${line:3}"
    	   folder="${folder##*[}"
    	   folder="${folder%%]*}"
	   continue

        # subheader
        elif [ "${line:0:3}" == "###" ]
        then
	   subfolder="${line:4}"
    	   subfolder="${subfolder##*[}"
    	   subfolder="${subfolder%%]*}"
    	   continue

        # author
        elif [ "${line:0:1}" == "*" ]
        then
	   name="${line%%,*}"
    	   name="${name##* }"

        # title
        else
	   title=$line
    	   keyword="${title%% *}"
	   keyword="${keyword%%:}"
    	   continue

        fi

        path="./$folder/$subfolder"
        newname="[2017 $name] $title.pdf"
        file="$name"_"$keyword"*_CVPR_2017_paper.pdf
        printf "$file\n"
        if ls ./$file 1> /dev/null 2>&1
        then
        	mkdir -p "$path"
        	mv ./$file "$path/$newname"
        fi
    fi
done <README.md



