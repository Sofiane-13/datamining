polyreg <- function(){
  
  d = read.table("1000.data",header=T);

  plot(d$x,d$y)
  for(i in 1:10){
    n = d[sample(1000),];
    x = d[,1]
    y = d[,2]
  apprentissage=n[1:20,]
  model <- lm(y ~ poly(x,5),apprentissage)
  copieData=d
  copieData$pred <- predict(model, data.frame(x=copieData$x))

  line=lines(copieData$x,copieData$pred, col=rainbow(10)[i])
  }
  return(line)

}
polyreg()