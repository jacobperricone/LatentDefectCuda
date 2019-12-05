for f in *.txt; 
do awk -F'[." "]' '{for(i=1;i<=NF;i+=2){printf $i " " }}{printf "\n"}' $f  > tmp && mv tmp $f
done