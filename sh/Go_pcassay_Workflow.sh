#!/bin/bash
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
	--o ${DATADIR}/aid_${AID}_substances.tsv
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
#
# Columns: SMILES, CID
python3 -m BioClients.pubchem.Client get_cid2smiles \
	--i ${DATADIR}/aid_${AID}_compounds.cid \
	--o ${DATADIR}/aid_${AID}_compounds.smiles
#
# Columns: 
python3 -m BioClients.pubchem.Client get_cid2properties \
	--i ${DATADIR}/aid_${AID}_compounds.cid \
	--o ${DATADIR}/aid_${AID}_compounds_properties.tsv
#
python3 -m BioClients.util.pandas.Utils showcols --i ${DATADIR}/aid_${AID}_compounds_properties.tsv
python3 -m BioClients.util.pandas.Utils summary --i ${DATADIR}/aid_${AID}_compounds_properties.tsv
#
###
# RDKit
###
