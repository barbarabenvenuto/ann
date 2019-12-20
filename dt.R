# labeling the objects and fixing the column names

dpath = "~/ann/anndataset_n.txt"
dataset = read.delim(dpath)
colnames(dataset) = c('F378-r','F395-r','F410-r','F430-r','F515-r','r-F660','r-F861','g-r','r-i','r-r','u-r','r-z')
label = row.names(dataset)
for (i in 1:length(label)) {label[i] = ifelse(substr(label[i],1,1) == "t", 2, 1)}  # 1 ssp, 2 star
label = as.integer(label)
dataset = cbind(dataset,label)

dataset1 = data.frame(cbind(dataset$label,dataset$`u-r`,dataset$`F378-r`,dataset$`F395-r`,dataset$`F410-r`,dataset$`F430-r`,
                            dataset$`g-r`,dataset$`F515-r`,dataset$`r-F660`,dataset$`r-i`,dataset$`r-F861`,dataset$`r-z`))
colnames(dataset1) = c('label','u-r','F378-r','F395-r','F410-r','F430-r','g-r','F515-r','r-F660','r-i','r-F861','r-z')
row.names(dataset1) = row.names(dataset)

write.table(dataset1, file="anndataset_n_l.txt")  # normalized, labeled
