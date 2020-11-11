#/bin/sh

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=~/nematus/data/mosesdecoder

# suffix of target language files
lng=de

sed 's/\@\@ //g' | \
perl ./src/metric/scripts/recaser/detruecase.perl |
perl ./src/metric/scripts/tokenizer/detokenizer.perl -l de