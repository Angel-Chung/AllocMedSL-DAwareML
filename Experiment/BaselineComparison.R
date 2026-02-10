library(data.table)
library(stringr)
library(dplyr)
library(ggplot2)
library("viridis")  
options(scipen=999)

setwd("")

# Decision-Aware Prior
files <- list.files(path = "/Experiment/DecisionAware/result/", pattern = "*.csv")
data_list=list()
for (i in 1:length(files)) {
  data <- fread(paste0("/Experiment/DecisionAware/result/", files[i]), header = TRUE)
  data_list[[i]] <- data  # Use 'i' as the index
}
DA <- rbindlist(data_list)
DA$Approach <- "Decision-Aware (w/ Prior)"

EM <- DA
EM %>%
  dplyr::select(date,hf_pk,product,allocation,Approach,target,stock,district)%>%
  rename(districtID=district)-> EM
EM$unmetDemand <- ifelse(EM$target-EM$allocation<0,0,EM$target-EM$allocation)
EM <- as.data.table(EM)

plotDAPrior_Real25 <- EM[,.(unmetDemand=sum(unmetDemand,na.rm=TRUE),target=sum(target,na.rm=TRUE)), by=.(date,product,Approach)]
plotDAPrior_Real25$unmetDemand_pct <- (plotDAPrior_Real25$unmetDemand/plotDAPrior_Real25$target)*100

# Decision-Blind
files <- list.files(path = "/Experiment/DecisionBlind/result/", pattern = "*.csv")
data_list=list()
for (i in 1:length(files)) {
  data <- fread(paste0("/Experiment/DecisionBlind/result/", files[i]), header = TRUE)
  data_list[[i]] <- data  # Use 'i' as the index
}
DB<- rbindlist(data_list)
DB$Approach <- "Decision-Blind"

EM <- DB
EM %>%
  dplyr::select(date,hf_pk,product,allocation,Approach,target,district)%>%
  rename(districtID=district)-> EM
EM$unmetDemand <- ifelse(EM$target-EM$allocation<0,0,EM$target-EM$allocation)
EM <- as.data.table(EM)

plotDB_Real25 <- EM[,.(unmetDemand=sum(unmetDemand,na.rm=TRUE),target=sum(target,na.rm=TRUE)), by=.(date,product,Approach)]
plotDB_Real25$unmetDemand_pct <- (plotDB_Real25$unmetDemand/plotDB_Real25$target)*100

# Distribution
files <- list.files(path = "/Experiment/Distribution/results/", pattern = "*.csv")
data_list=list()
for (i in 1:length(files)) {
  data <- fread(paste0("/Experiment/Distribution/results/", files[i]), header = TRUE)
  data_list[[i]] <- data  # Use 'i' as the index
}
d <- rbindlist(data_list)

d$Approach <- "Distribution Modeling"
EM <- d
EM %>%
  select(date,hf_pk,product,target,allocation,Approach)-> EM
EM$unmetDemand <- ifelse(EM$target-EM$allocation<0,0,EM$target-EM$allocation)
plotDM_Real25 <- EM[,.(unmetDemand=sum(unmetDemand,na.rm=TRUE),target=sum(target,na.rm=TRUE)), by=.(date,product,Approach)]
plotDM_Real25$unmetDemand_pct <- (plotDM_Real25$unmetDemand/plotDM_Real25$target)*100

# Global Health (3mth Avg)
files <- list.files(path = "/Experiment/Global Health (3 Month Rolling Avg)/results/", pattern = "*.csv")
data_list=list()
for (i in 1:length(files)) {
  data <- fread(paste0("/Experiment/Global Health (3 Month Rolling Avg)/results/", files[i]), header = TRUE)
  data_list[[i]] <- data  # Use 'i' as the index
}
gb <- rbindlist(data_list)
gb$Approach <- "Global_Health (3mth Avg)"
gb%>%
  dplyr::select(date,hf_pk,product,allocation,Approach,target)-> gb
gb$unmetDemand <- ifelse(gb$target-gb$allocation<0,0,gb$target-gb$allocation)
gb <- as.data.table(gb)
plotgb_Real25 <- gb[,.(unmetDemand=sum(unmetDemand,na.rm=TRUE),target=sum(target,na.rm=TRUE)), by=.(date,product,Approach)]
plotgb_Real25$unmetDemand_pct <- (plotgb_Real25$unmetDemand/plotgb_Real25$target)*100

# StochOptForest
files <- list.files(path = "/Experiment/StochOptForest/result", pattern = "*.csv")
data_list=list()
for (i in 1:length(files)) {
  data <- fread(paste0("/Experiment/StochOptForest/result", files[i]), header = TRUE)
  data_list[[i]] <- data  # Use 'i' as the index
}
stoch <- rbindlist(data_list)
stoch$Approach <- "StochOptForest"
stoch  %>%
  dplyr::select(date,hf_pk,product,allocation,Approach,target)-> stoch 

stoch$unmetDemand <- ifelse(stoch$target-stoch$allocation<0,0,stoch$target-stoch$allocation)
stoch <- as.data.table(stoch)
plotStoch_Real25 <- stoch[,.(unmetDemand=sum(unmetDemand,na.rm=TRUE),target=sum(target,na.rm=TRUE)), by=.(date,product,Approach)]
plotStoch_Real25$unmetDemand_pct <- (plotStoch_Real25$unmetDemand/plotStoch_Real25$target)*100


# Population based
map <- fread("/Experiment/Population Based/ChiefdomPopMapUpdate.csv")
facility <- fread("/Experiment/S1_master_facility_update_11.csv")
facility%>%
  select(hf_pk,chiefdom,district)%>%
  left_join(map) -> facility
facility <- as.data.table(facility)
facility <- facility[district==Dist | Dist==1,]

facility %>%
  select(district,GEO2_SL2015,chiefdom)%>%
  unique() -> disP

facility <- left_join(facility,disP)
facility$GEO2_SL2015 <- ifelse(is.na(facility$GEO2_SL2015),facility$distPop,facility$GEO2_SL2015)
catchpop <- fread("/Experiment/Population Based/sateliePop.csv")
catchpop %>%
  rename(hf_pk=id)%>%
  select(hf_pk,friction_catchment_population) -> catchpop
our <- DA
our%>% select(date,hf_pk,product,allocation,target,stock) -> our
our <- left_join(our,facility)
our %>%
  select(product,hf_pk,date,target,district,stock)%>%
  rename(districtID=district)%>%
  left_join(facility)%>%
  left_join(catchpop)%>%
  group_by(product,date)%>%
  mutate(totalP=sum(friction_catchment_population,na.rm=T))%>%
  mutate(PopProp=(friction_catchment_population/totalP))%>%
  mutate(date=as.character(date))-> PopPure

PopPure %>%
  group_by(product,date)%>%
  mutate(allocation=PopProp*stock) -> PopPure

PopPure$Model <- "Population Based Census"
PopPure$Approach <- "Population Based Census"
PopPure %>%
  dplyr::select(hf_pk,product,target,allocation,Model,Approach,stock)-> PopPure

PopPure$unmetDemand <- ifelse(PopPure$target-PopPure$allocation<0,0,PopPure$target-PopPure$allocation)
PopPure <- as.data.table(PopPure)
plotPop_Real25 <- PopPure[,.(unmetDemand=sum(unmetDemand,na.rm=TRUE),target=sum(target,na.rm=TRUE)), by=.(date,product,Approach)]
plotPop_Real25$unmetDemand_pct <- (plotPop_Real25$unmetDemand/plotPop_Real25$target)*100
plotPop_Real25$date <- as.IDate(plotPop_Real25$date)

# Excel 
Excel <- fread("/Experiment/Excel/ExcelQ2.csv") #NMSA Q2 Excel Allocation shared by local staff 
Excel <- Excel[!is.na(Excel$product),]
long <- melt(setDT(Excel), id.vars = c("product","name"), variable.name = "district")
long <- rename(long, districtExcel=district)
Exceldist <- fread("/Experiment/Excel/Exceldistrict.csv")
long <- left_join(long,Exceldist)
long <- long[,.(allocation=sum(value,na.rm=T)),by=.(product,district)]
long <- left_join(long,Target)
long$unmetDemand <- ifelse(long$target-long$allocation<0,0,long$target-long$allocation)
long$unmetDemand_pct <- (long$unmetDemand/long$target)*100
ExcelPerf <- long[,.(unmetDemand_pct =mean(unmetDemand_pct ,na.rm=T)),by=.(product)]
mean(ExcelPerf$unmetDemand_pct)
