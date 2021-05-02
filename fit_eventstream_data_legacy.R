library(yaml)
library(dplyr)
library(bdmiso)
library(bdglm)
library(splines)
library(optparse)


# Parsing
parser <- OptionParser()
parser <- add_option(parser, c("--infolder"), type="character", help="Location of datasets")
parser <- add_option(parser, c("--outfolder"), type="character", help="Location to store output")
parser <- add_option(parser, c("--seed"), type="integer", help="RNG Seed")
parser <- add_option(parser, c("--name"), type="character", help="Name to read in and write")
parser <- add_option(parser, c("--fineness"), type="double", help="Level of discretization to use")
parser <- add_option(parser, c("--verbose", "-v"), action="store_true")
args <- parse_args(parser)

# Iterated over specified files
inpath <- path.expand(args$infolder)
for(file in dir(inpath)){
  if(startsWith(x = file, prefix = args$name)){
    infile <- file.path(inpath, file)
    print(paste0("Reading ", infile))
    dat <- read_yaml(infile)
    fittime_start <- proc.time()[3]
    fineness <- args$fineness
    timeticks <- seq(0, dat$maxtime, by=fineness)
    mem <- dat$breakpoints[length(dat$breakpoints)]/fineness
    breakpoints <- dat$breakpoints
    
    mspikes_o <- miso.spikes(dat$output, timeticks)
    mspikes <- lapply(1:dat$nsources, function(i) miso.spikes(dat[[paste0("source_", i)]], timeticks))
    spiketrain <- cbind(bind_cols(mspikes), mspikes_o)
      
    ninput <- dat$nsources + 1
    names(spiketrain)=c("response", paste0("source_",1:dat$nsources))
    
    knots <- breakpoints[-c(1, length(breakpoints))] # code auto-adds breakpoints at 0 and mem.
    
    model.o1=miso.bspl.matrix.o1(spiketrain,mem,knots,degree = dat$order -1,output.index=ninput, d=fineness) 
    y=as.numeric(miso.spikes(dat$output,timeticks,type="binary")[-c(1:(mem-1))])
    y <- y[1:nrow(model.o1$x)]
    
    control <- bdglm.control()
    x1 <- cbind(1, model.o1$x)
    start = rep(0, ncol(x1))
    print(paste0("Fitting ", infile))
    res <- bdglm.fit(x1, y, family=binomial("logit"), start=start, intercept=FALSE, control=control)
    fittime_end <- proc.time()[3]
    fittime <- fittime_start - fittime_end * 1000 #defaults to seconds.
    # save output
    out <- list(fineness=fineness, coefs = coef(res), fittime=fittime)
    outfile <- file.path(path.expand(args$outfolder), file)
    write_yaml(out, outfile)
  }
}
