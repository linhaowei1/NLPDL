rm -rf Task1.zip Task2.zip Assignment_2.zip
zip -r Task1.zip dependency_parsing/*.py dependency_parsing/data dependency_parsing/utils
zip -r Task2.zip nmt/*.py nmt/chr_en_data nmt/sanity_check_en_es_data nmt/outputs
zip -r Assignment_2.zip Task1.zip Task2.zip
rm -rf Task1.zip Task2.zip
