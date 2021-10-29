#!/bin/bash
###
# Query PubChem for a specific bioassay, get compounds and bioactivity
# results (Active/Inactive).
# Generate fps into ML-ready X vectors, with bioactivity as Y labels for
# binary classification ML.
###
#
cwd=$(pwd)
#
DATADIR="${cwd}/data"
#
AID=527
#
NAME=$(python3 -m BioClients.pubchem.Client get_assayname --aid ${AID} |awk -F '\t' '{print $2}')
printf "Assay name: \"${NAME}\"\n"
#
python3 -m BioClients.pubchem.Client get_assaysubstances \
	--aid ${AID} --o ${DATADIR}/aid_${AID}_substances.tsv
cat ${DATADIR}/aid_${AID}_substances.tsv |awk -F '\t' '{print $2}' |sed '1d' \
	>${DATADIR}/aid_${AID}_substances.sid
printf "Substances: %d\n" $(cat ${DATADIR}/aid_${AID}_substances.sid |wc -l)
cat ${DATADIR}/aid_${AID}_substances.tsv |awk -F '\t' '{print $3}' |sed '1d' |sort -nu \
	>${DATADIR}/aid_${AID}_compounds.cid
printf "Compounds: %d\n" $(cat ${DATADIR}/aid_${AID}_compounds.cid |wc -l)
#
python3 -m BioClients.pubchem.Client get_assaysubstanceresults \
	--aid ${AID} --i ${DATADIR}/aid_${AID}_substances.sid \
	--o ${DATADIR}/aid_${AID}_results.tsv
python3 -m BioClients.util.pandas.Utils list_columns --i ${DATADIR}/aid_${AID}_results.tsv
python3 -m BioClients.util.pandas.Utils colvalcounts \
	--coltags "Bioactivity Outcome" \
	--i ${DATADIR}/aid_${AID}_results.tsv
#
# Columns: CID,CanonicalSMILES,IsomericSMILES
python3 -m BioClients.pubchem.Client get_cid2smiles \
	--i ${DATADIR}/aid_${AID}_compounds.cid \
	--o ${DATADIR}/aid_${AID}_compounds.smiles
python3 -m BioClients.util.pandas.Utils list_columns --i ${DATADIR}/aid_${AID}_compounds.smiles
#
# Columns: 
python3 -m BioClients.pubchem.Client get_cid2properties \
	--i ${DATADIR}/aid_${AID}_compounds.cid \
	--o ${DATADIR}/aid_${AID}_compounds_properties.tsv
#
python3 -m BioClients.util.pandas.Utils list_columns --i ${DATADIR}/aid_${AID}_compounds_properties.tsv
#
###
# RDKit
###
source $(dirname $(which conda))/../bin/activate rdkit
#
python3 -m rdktools.fp.App FingerprintMols \
	--i ${DATADIR}/aid_${AID}_compounds.smiles \
	--smilesColumn 1 --nameColumn 0 \
	--fpAlgo MACCS --output_as_dataframe \
	--o ${DATADIR}/aid_${AID}_compounds_MaccsFps.pkl
#
source $(dirname $(which conda))/../bin/deactivate
###
# Next: Scikit-learn binary classification ML.
