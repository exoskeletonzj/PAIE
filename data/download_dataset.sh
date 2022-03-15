# Download RAMS
wget -c https://nlp.jhu.edu/rams/RAMS_1.0b.tar.gz
tar -zxvf ./RAMS_1.0b.tar.gz
rm -rf ./RAMS_1.0b.tar.gz
mv ./RAMS_1.0 ./data/

# Download WIKIEVENTS
mkdir -p ./data/WikiEvent/
wget -c -P ./data/WikiEvent https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/train.jsonl
wget -c -P ./data/WikiEvent https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/dev.jsonl
wget -c -P ./data/WikiEvent https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/test.jsonl