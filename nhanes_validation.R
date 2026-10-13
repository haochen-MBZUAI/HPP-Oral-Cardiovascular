#!/usr/bin/env Rscript
# Inputs: candidate CSV, NHANES metadata CSV, complete raw genus counts CSV, output CSV.
args <- commandArgs(trailingOnly=TRUE)
suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu="adjust")
candidates <- read.csv(args[1],check.names=FALSE)
meta <- read.csv(args[2],check.names=FALSE)
raw <- read.csv(args[3],check.names=FALSE)
ids <- raw$sample_id
counts <- as.matrix(raw[,setdiff(names(raw),"sample_id"),drop=FALSE])
counts <- counts[,colSums(counts>0)>=200,drop=FALSE]
replaced <- counts
for (g in colnames(counts)) replaced[counts[,g]==0,g] <- quantile(counts[counts[,g]>0,g],.01,names=FALSE)/2
logppm <- log10(replaced/rowSums(replaced)*1e6)
if ("WTMEC2YR" %in% names(meta)) meta$weight_4yr <- meta$WTMEC2YR/2
bp <- intersect(paste0("BPXSY",1:4),names(meta))
if (length(bp)) {
  readings <- as.matrix(meta[,bp,drop=FALSE]); readings[readings<=0] <- NA_real_
  meta$sbp <- rowMeans(readings,na.rm=TRUE); meta$sbp[!is.finite(meta$sbp)] <- NA_real_
}
fields <- c("age","sex","smoking_status","weight_4yr","psu","strata")
meta <- meta[complete.cases(meta[,fields]) & meta$age>=18 & meta$weight_4yr>0 & (!is.na(meta$sbp)|!is.na(meta$pulse)),]
meta$sex <- factor(meta$sex)
meta$smoking_status <- factor(meta$smoking_status,levels=c("Never","Former","Current"))
sets <- unique(candidates[,c("genus","endpoint"),drop=FALSE])
rows <- lapply(seq_len(nrow(sets)),function(i) {
  g <- sets$genus[i]; y <- sets$endpoint[i]
  result <- data.frame(genus=g,endpoint=y,N=0,beta=NA_real_,raw_P=NA_real_)
  if (!g %in% colnames(logppm)) return(result)
  d <- merge(meta,data.frame(sample_id=ids,x=logppm[,g],detected=counts[,g]>0),by="sample_id")
  d <- d[complete.cases(d[,c(y,"x",fields)]),]
  result$N <- nrow(d)
  if (nrow(d)<100 || sum(d$detected)<200 || sd(d$x)==0) return(result)
  d$x <- as.numeric(scale(d$x))
  design <- svydesign(ids=~psu,strata=~strata,weights=~weight_4yr,nest=TRUE,data=d)
  fit <- svyglm(as.formula(paste(y,"~ x + age + sex + smoking_status")),design=design)
  coefficient <- summary(fit)$coefficients["x",]
  result$beta <- unname(coefficient[1]); result$raw_P <- unname(coefficient[4])
  result
})
result <- if (length(rows)) do.call(rbind,rows) else data.frame(genus=character(),endpoint=character(),N=integer(),beta=double(),raw_P=double())
write.csv(result,args[4],row.names=FALSE)
