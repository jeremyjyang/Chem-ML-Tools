#!/bin/bash
#
cwd=$(pwd)
#
DATADIR="${cwd}/data"
#
source $(dirname $(which conda))/../bin/activate rdkit
#
python3 -m rdktools.fp.Extras list_maccskeys \
	--o $DATADIR/rdkit_maccskeys.tsv
#
python3 -m rdktools.depict.App pdf \
	--i $DATADIR/rdkit_maccskeys.tsv \
	--header --smilesColumn 2 --nameColumn 0 \
	--ifmt SMI --parse_as_smarts \
	--pdf_doctype "legalpaper" --pdf_landscape \
	--nPerRow 9 --nPerCol 4 \
	--o $DATADIR/rdkit_maccskeys.pdf
#
conda deactivate
#
