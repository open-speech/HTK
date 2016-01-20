# Usage: awk -f mlflabs2tgtscp labfile a=1 scpfile
{
    if (a==0) {
        # MLF filenames
	cnt++
	labstg[cnt]=sprintf("%s",substr($1,2,length($1)-6));
	labmark[labstg[cnt]]=1;
        #if ((substr($1,1,1) == "\"")) {
        #    n=split($1,b,"/");      
        #    cnt++;
        #    labstg[cnt] = sprintf("%s",substr(b[n],1,length(b[n])-5));
        #    labmark[labstg[cnt]] = 1;
	#}
    } else {
        # scp
	n=split($1,b,".");
	srcstg = sprintf("%s",substr($1,1,length($1)-4));
	if(labmark[srcstg]==1){
	    print $0
	}
	#n=split($1,b,".");
	#n=split(b[1],c,"/");
        #srcstg = c[1];
        #if (labmark[srcstg] == 1) {
        #    print $0;
        #}
    }
}
