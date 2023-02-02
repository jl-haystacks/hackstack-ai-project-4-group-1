# rm data/raw/covid-19-pandemic-worldwide-data.csv
rm data/pickle/ga_info.p

# wget -O data/raw/covid-19-pandemic-worldwide-data.csv "https://public.opendatasoft.com/explore/dataset/covid-19-pandemic-worldwide-data/download/?format=csv&timezone=Europe/Berlin&lang=fr&use_labels_for_header=true&csv_separator=%3B"

cd scripts && echo ls && python create_ga_fig.py
cd ..
python haystacks_prototype_map.py