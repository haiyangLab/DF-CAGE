require(optparse)
require(graphics)
require(mice)
options(stringsAsFactors=FALSE) 


option_list = list(
  make_option(c("-c", "--path"), action="store", default='./input/original.csv', 
              type='character', help="back_table"),
  make_option(c("-s", "--score_path"), action="store", default='./input/jieguo1.csv', 
              type='character', help="back_table")
)
opt = parse_args(OptionParser(option_list=option_list))
df <- read.csv(opt$path,sep=",",header=TRUE,encoding="UTF-8")
head(df[,c(24:35)], 3)
mice_imputes = mice(df[,c(24:35)], m=5, maxit = 40, method='pmm', seed=500)
Imputed_df=complete(mice_imputes,5)
df[,c(24:35)] <- Imputed_df
write.table(df, file = opt$score_path,row.names=FALSE,col.names=TRUE, sep=",",quote = FALSE)
