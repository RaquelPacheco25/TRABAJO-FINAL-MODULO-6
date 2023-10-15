
## TRABAJO FINAL SVM 

## Cargando librerias 

library(foreign)
library(dplyr)
library(caret)
library(ROCR)
library(ggplot2)
library(lattice)
library(e1071)


## Cargando la base 

datos<- read.spss("C:\\Users\\CompuStore\\Desktop\\CURSOS\\CIENCIA DE DATOS\\MODULO 6\\MACHINE LEARNING 1\\ENCUESTA NACIDOS VIVOS\\ENV_2017.sav",
                  use.value.labels = FALSE, to.data.frame = TRUE)

# Depurar la informacion 

table(datos$prov_nac)
str(datos$prov_nac)

datos$prov_nac<-as.numeric(as.character(datos$prov_nac))

nuevadata<-datos%>%
  filter(prov_nac==13)%>%
  select(peso, talla, sem_gest, sexo, edad_mad, con_pren, sabe_leer)%>%
  filter(peso!=99,
         talla!=99,
         sem_gest!=99,
         sabe_leer!=9,
         con_pren!=99)%>%
  mutate(peso=ifelse(peso>2500,1,0),
         sexo=ifelse(sexo==1,0,1),
         sabe_leer=ifelse(sabe_leer==1,1,0),
         con_pren=ifelse(con_pren>=7,1,0),
         edad2=edad_mad^2)


nuevadata$peso<-factor(nuevadata$peso)
nuevadata<- nuevadata%>%
  mutate(peso=recode_factor(
    peso,'0'="no adecuado",
    '1'="adecuado"))
table(nuevadata$peso)


# Fijar una semilla 

set.seed(1234)

# Crear una muestra de entrenamiento 

entrenamiento<-createDataPartition(nuevadata$peso, p=0.1, list = F)


# Realizamos el modelo SVM con la muestra de entrenamiento 

modeloSVM<- svm(peso ~ talla+sem_gest+sexo+edad_mad+
                  edad2+sabe_leer, data=nuevadata[entrenamiento,], 
                kernel="linear", cost=10, scale = T, probability=TRUE)
summary(modeloSVM)

## ¿Cómo recupero los vectores de soporte?

modeloSVM$index

## ¿Como recuepero el termino independiente?

modeloSVM$rho

## ¿Como recupero coeficientes que usan para multiplicar para cada observacion 
# y obtener el vector perpendicular al plano 

modeloSVM$coefs


#### Evaluar el modelo 

ajustados <- predict(modeloSVM, nuevadata[entrenamiento,],
                     type="prob")

# Por defecto se clasifica con un corte de 0.5

# Forma larga de matriz de clasificacion 
# Matriz de confusion 

ct<- table(nuevadata[entrenamiento,]$peso, 
           ajustados, dnn=c("Actual", "Predicho"))
diag(prop.table(ct,1))
sum(diag(prop.table(ct)))

confusionMatrix(nuevadata$peso[entrenamiento],
                ajustados, dnn=c("Actual", "Predicho"),
                levels(ajustados)[2])
plot(modeloSVM, data=nuevadata[entrenamiento,],
     talla ~ sem_gest)

### Optimizar o tunear nuestro modelo 

modelo_tuneado<-tune(svm, peso~.,
                     data=nuevadata[entrenamiento,],
                     ranges= list(cost=c(0.001, 0.01, 0.1, 1,5,10, 50)),
                     kernel="linear", scale=TRUE, probability=TRUE, )
summary(modelo_tuneado)

ggplot(data=modelo_tuneado$performances, 
       aes(x=cost, y=error))+
  geom_line()+
  geom_point()+
  labs(title = "Error de validacion vs Hiperparametro C")+
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5))


mejor_modelo<- modelo_tuneado$best.model
summary(mejor_modelo)


## Vectores de soportes

head(mejor_modelo$index, 100)

plot(mejor_modelo, data=nuevadata[entrenamiento,],
     talla~sem_gest)


## Validando el "mejor modelo" 

ajustados_mejor_modelo <- predict(mejor_modelo, nuevadata[entrenamiento,],
                                  type="prob", probability = TRUE)

#verificar como captura el modelo 

str(ajustados_mejor_modelo)
head(attr(ajustados_mejor_modelo, "probabilities"),5)

# Matriz de confusion o clasificacion 

# verificar cual es la primera
# que arroja el modelo tuneado 
# en base a esto apuntar al vector de probabilidades y realizar puebas correctas

levels(ajustados_mejor_modelo)

table(attr(ajustados_mejor_modelo,"probabilities")[,1]>0.5,
      nuevadata$peso[entrenamiento])

levels(nuevadata$peso)

confusionMatrix(ajustados_mejor_modelo, nuevadata$peso[entrenamiento],
                positive = levels(nuevadata$peso)[2])

# Analizar las curvas ROC

pred<-prediction(attr(ajustados_mejor_modelo, "probabilities")[,2], 
                 nuevadata$peso[entrenamiento])

perf<-performance(pred, "tpr", "fpr")
plot(perf, colorize=T, lty=3)
abline(0,1, col="black")

# Area bajo la curva

aucemodelo1<- performance(pred, measure = "auc")
aucemodelo1<-aucemodelo1@y.values[[1]]
aucemodelo1

# Sensitividad y especificidad 

plot(performance(pred, measure = "sens",
                 x.measure = "spec", colorize=T))


## Punto de corte optimo 
# Enfoque maximizacion sensitividad y especificidad 

perf1<- performance(pred, "sens", "spec")
sen<- slot(perf1, "y.values"[[1]])
esp<-slot(perf1, "x.values"[[1]])
alf<-slot(perf1, "alpha.values"[[1]])
mat<-data.frame(alf,sen,esp)

names(mat)[1]<-"alf"
names(mat)[2]<-"sen"
names(mat)[3]<-"esp"
library(reshape2)
library(ggplot2)
library(plotly)

m<-melt(mat, id=c("alf"))

p1 <- ggplot(m, aes(alf, value, group=variable, colour=variable)) +
  geom_line(size=1.2) +
  labs(title = "punto de corte para svm", x="cut-off", y="")

ggplotly(p1)

## otro enfoque para el cut-off
## determinar el cut-off que maximiza el acurracy de mi modelo 

max.acurracy<-performance(pred, measure = "acc")
plot(max.acurracy)

indice<-which.max(slot(max.acurracy, "y.values")[[1]])
acc<-slot(max.acurracy, "y.values")[[1]][indice]
cutoff<-slot(max.acurracy, "x.values")[[1]][indice]
print(c(accuracy=acc, 
        cutoff=cutoff))

# Otro enfoque 
library(pROC)

prediccionescutoff<-attr(ajustados_mejor_modelo, "probabilities")[,1]

curvaROC<-plot.roc(nuevadata$peso[entrenamiento],
                   as.vector(prediccionescutoff),
                   precent=TRUE, ci=TRUE, print.auc=TRUE,
                   threholds="best",
                   print.thres="best")




## CON EL PUNTO OPTIMO

# Matriz de confusion o clasificacion 

levels(ajustados_mejor_modelo)

table(attr(ajustados_mejor_modelo,"probabilities")[,1]>0.78,
      nuevadata$peso[entrenamiento])

levels(nuevadata$peso)

confusionMatrix(ajustados_mejor_modelo, nuevadata$peso[entrenamiento],
                positive = levels(nuevadata$peso)[2])

# Analizar las curvas ROC

pred2<-prediction(attr(ajustados_mejor_modelo, "probabilities")[,2], 
                 nuevadata$peso[entrenamiento])

perf<-performance(pred2, "tpr", "fpr")
plot(perf, colorize=T, lty=3)
abline(0,1, col="black")

# Area bajo la curva

aucemodelo2<- performance(pred, measure = "auc")
aucemodelo2<-aucemodelo2@y.values[[1]]
aucemodelo2

# Sensitividad y especificidad 

plot(performance(pred, measure = "sens",
                 x.measure = "spec", colorize=T))


##Punto de corte 0.78

## otro enfoque para el cut-off
## determinar el cut-off que maximiza el acurracy de mi modelo 

max.acurracy2<-performance(pred2, measure = "acc")
plot(max.acurracy2)

indice2<-which.max(slot(max.acurracy2, "y.values")[[1]])
acc2<-slot(max.acurracy2, "y.values")[[1]][indice2]
cutoff2<-slot(max.acurracy2, "x.values")[[1]][indice2]
print(c(accuracy=acc2, 
        cutoff=cutoff2))

# Otro enfoque 
library(pROC)

prediccionescutoff2<-attr(ajustados_mejor_modelo, "probabilities")[,1]

curvaROC2<-plot.roc(nuevadata$peso[entrenamiento],
                   as.vector(prediccionescutoff2),
                   precent=TRUE, ci=TRUE, print.auc=TRUE,
                   threholds="best",
                   print.thres="best")



## Prediciendo con SVM

newdata<-head(nuevadata,5)
str(newdata)

# Predecir dentro de la muestra
#Considerar que para la prediccion el punto de corte por defecto es 0,5

predict(mejor_modelo,newdata)
pronostico1<-predict(mejor_modelo, newdata)
pronostico1

p.probabilidades<-predict(mejor_modelo, newdata, probability = TRUE)
p.probabilidades


## Pronostico fuera de la muestra

newdata2<-data.frame(
  talla=47,
  sem_gest=37,
  sexo=1,
  edad_mad=25,
  sabe_leer=1,
  con_pren=2,
  edad2=50)

pronostico2<-predict(mejor_modelo, newdata2, probability = TRUE)
pronostico2

predict(mejor_modelo, newdata2)

# Evaluando punto de corte sugerido 
#Definamos el punto de corte

umbral<-as.numeric(cutoff)
umbral

table(attr(ajustados_mejor_modelo,"probabilities")[,1]>umbral,
      nuevadata$peso[entrenamiento])

# Echar un vistazo de las probabilidades devueltas

head(attr(ajustados_mejor_modelo, "probabilities"))

# Seleccionamos la probabilidad objetivo 

prediccionescutoff<-attr(ajustados_mejor_modelo, "probabilities")[,1]
prediccionescutoff<-as.numeric(prediccionescutoff)


predCut<-factor(ifelse(prediccionescutoff>umbral,1,0))

matrizpuntocorte<- data.frame(real=nuevadata$peso[entrenamiento],
                              predicho=predCut)

matrizpuntocorte<-matrizpuntocorte%>%
  mutate(predicho=recode_factor(predicho,'0'="no adecuado",
                                '1'="adecuado"))

# Calcula la matriz de confusión
confusionMatrix(matrizpuntocorte$predicho, matrizpuntocorte$real, positive = "adecuado")

library(ROSE)


#### CUANDO LOS DATOS ESTAN DESBALANCEADOS 

train_data<- nuevadata[entrenamiento,]
table(train_data$peso)


## ROSE:metodo sintético para completar la informacion mediante una densidad de kernel

# se supone que es el metodo mas robusto

roses<- ROSE(peso~.,
             data=train_data,
             seed = 1)$data

table(roses$peso)

# Corramos el modelo con las 3 tecnicas de remuestreo
# desbalance muestral 


modelo.rose<-tune(svm, peso~.,
                  data=roses, 
                  ranges= list(cost=c(0.001, 0.01, 0.1, 1,5,10, 50)),
                  kernel="linear", scale=TRUE, probability=TRUE)
summary(modelo.rose)
mejor.modelo.rose<- modelo.rose$best.model

### Evaluacion del modelo re muestreo 

ajustadosrose<- predict(mejor.modelo.rose, 
                        roses, 
                        type="prob",
                        probability = T)
# CARET 

confusionMatrix(roses$peso, ajustadosrose, 
                dnn=c("Actuales", "Predichos"),
                levels(ajustadosrose)[1])


confusionMatrix(ajustados_mejor_modelo,nuevadata$peso[entrenamiento], 
                positive=levels(nuevadata$peso)[2])


# Curvas ROC


predrose<-prediction(attr(ajustadosrose, 
                          "probabilities")[,2],
                     roses$peso)


roc.curve(roses$peso, 
          attr(ajustadosrose, 
               "probabilities")[,2],
          col="green", add.roc = T)

roc.curve(nuevadata$peso[entrenamiento], 
          attr(ajustados_mejor_modelo, 
               "probabilities")[,2],
          col="pink", add.roc = T)



## Pronostico 

newdata3<-data.frame(
  talla=58,
  sem_gest=22,
  sexo=2,
  edad_mad=35,
  sabe_leer=1,
  con_pren=1,
  edad2=70)

pronostico3<-predict(mejor.modelo.rose, newdata3, probability = TRUE)
pronostico3

### DATAFRAME Pronostico 

finalpronostico<- data.frame(pronostico2, pronostico3)
finalpronostico

