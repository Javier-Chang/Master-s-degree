#setwd("D:/time")/
co = read.csv("D:/co.csv") 
library(caret)
library(forecast)
ptime <- ts(co, frequency=12, start=c(2008,1))
fit <- nnetar(ptime,repeats=40)
fcast <- forecast(fit)
plot.forecast(fcast)
lines(fit$x,col="red")
lines(fitted(fit),col="green")
legend("topright",lty=1,col=c(2,3,4),
       legend=c("observed","fit","forecast"))
summary(fcast)
Box.test(fit$residuals,lag=1,type="Ljung")


