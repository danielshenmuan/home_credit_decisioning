library(dplyr)
library(tidyverse)
library(class)
library(kknn)
library(gbm)
library(MLmetrics)
library(adabag)
library(rpart)
library(caret)
library(ROSE)
library(glmnet)
library(caret)
set.seed(1)

#Read Data
cr_data <- read.csv("application_train.csv",na.strings=c("NA","NaN","?", "","XNA"))
#==================================================================================================
#Data Cleaning
#==================================================================================================
#Get Mode Fucntion
getmode <- function(v) {
  uniqv <- unique(v[!is.na(v)])
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

## Remove columns with more than 60% NA
cr_data <-  cr_data[, which(colMeans(!is.na(cr_data)) >= 0.4)]

#Convert Target to factor
cr_data <- cr_data %>% mutate(TARGET = TARGET %>% as.character() %>% as.factor())

#Select String Columns
string_2_factor_names <- cr_data %>% select_if(is.character) %>% names()

#Impute Mode for String Cols
cr_data <- cr_data %>% mutate_at(vars(string_2_factor_names),~ifelse(is.na(.x), getmode(.x), .x))

#Convert String columns to factors
cr_data[string_2_factor_names] <- lapply(cr_data[string_2_factor_names], factor)


#Which numeric data should be factored (categorical). These typically have a low number of unique levels. 
#dplyr and purrr operations to get a count of the unique levels. 
uni_numeric_vals <- cr_data %>% select_if(is.numeric) %>%  map_df(~ unique(.) %>% length()) %>%
  gather() %>% arrange(value) %>% mutate(key = as_factor(key))

#Converting numerical values with less than 5 unique values to categorical/nominal variables
factor_limit <- 5
num_2_factor_names <- uni_numeric_vals %>% filter(value < factor_limit) %>% arrange(desc(value)) %>%
  pull(key) %>% as.character()
cr_data[num_2_factor_names] <- lapply(cr_data[num_2_factor_names], factor)

#Impute Values - Mean for numeric
cr_data <- cr_data %>% mutate_at(vars(num_2_factor_names),~ifelse(is.na(.x), mean(.x, na.rm = TRUE), .x))

#AMT_REQ Columns
AMT_Cols <- cr_data %>% select(starts_with("AMT_REQ")) %>% names()
cr_data <- cr_data %>% mutate_at(vars(AMT_Cols),~ifelse(is.na(.x), getmode(.x), .x))
cr_data[AMT_Cols] <- lapply(cr_data[AMT_Cols], factor)

#FLAG Columns
FLAG_Cols <- cr_data %>% select(starts_with("FLAG_DOCUMENT")) %>% names()
cr_data <- cr_data %>% mutate_at(vars(FLAG_Cols),~ifelse(is.na(.x), getmode(.x), .x))
cr_data[FLAG_Cols] <- lapply(cr_data[FLAG_Cols], factor)

#CNT_SOCIAL_CIRCLE Columns
CNT_Cols <- cr_data %>% select(contains("CNT")) %>% names()
cr_data <- cr_data %>% mutate_at(vars(CNT_Cols),~ifelse(is.na(.x), getmode(.x), .x))
cr_data[CNT_Cols] <- lapply(cr_data[CNT_Cols], factor)

#Double Columns
Dbl_Cols <- cr_data %>% select_if(is.double) %>% names()
cr_data <- cr_data %>% mutate_at(vars(Dbl_Cols),~ifelse(is.na(.x), mean(.x, na.rm = TRUE), .x))
cr_data$FONDKAPREMONT_MODE[is.na(cr_data$FONDKAPREMONT_MODE)] =
  #Observe Data
  str(cr_data)
#Check for NA
sum(is.na(cr_data))
table(is.na(cr_data))

#Doublecheck, make sure there are no NAs 
cr_data = cr_data[complete.cases(cr_data) == T,]

##To save time fitting models, we sampled 1/10 of the data##
save_time = sample(1:nrow(cr_data),nrow(cr_data)/10)
cr_data = cr_data[save_time,]

#==========================================================================================================
#Exploratory Data Analysis
#==========================================================================================================

#Color Coding============================================================================================
color_combo <- c("1" = "firebrick", "0" = "darkgoldenrod3")
single_color <- "firebrick"
single_color2 <- "darkgoldenrod3"
line_color <-  "black"
text_size = 5

#Target Analysis=======================================================================================
TotalRows = nrow(cr_data)

cr_data %>% group_by(TARGET) %>% summarise(Count = n()/nrow(cr_data)*100) %>% 
  arrange(desc(Count))%>% mutate(TARGET = reorder(TARGET,Count)) %>%
  ggplot(aes(x = TARGET,y = Count)) +
  geom_bar(stat='identity',fill= color_combo,color=line_color) +
  geom_text(aes(x = TARGET, y = 1, label = paste0("( ",round(Count,2)," %)",sep="")),
            hjust=0, vjust=.5, size = text_size, colour = 'white',
            fontface = 'bold') +
  
  labs(x = 'TARGET', 
       y = 'Percentage', 
       title = 'TARGET Analysis',
       subtitle = '0 - No Problem in Payment ; 1 - Problem in Payment') +
  
  coord_flip() 

#Target Analysis after sampling=========================================================================
cr_sampled <- ovun.sample(TARGET~., data = cr_data, method = "both")$data

cr_sampled %>% group_by(TARGET) %>% summarise(Count = n()/nrow(cr_sampled)*100) %>% 
  arrange(desc(Count))%>% mutate(TARGET = reorder(TARGET,Count)) %>%
  ggplot(aes(x = TARGET,y = Count)) +
  geom_bar(stat='identity',fill= color_combo,color=line_color) +
  geom_text(aes(x = TARGET, y = 1, label = paste0("( ",round(Count,2)," %)",sep="")),
            hjust=0, vjust=.5, size = text_size, colour = 'white',
            fontface = 'bold') +
  
  labs(x = 'TARGET', 
       y = 'Percentage', 
       title = 'TARGET Analysis - Post Sampling',
       subtitle = '0 - No Problem in Payment ; 1 - Problem in Payment') +
  
  coord_flip() 

#Gender Analysis===========================================================================================
cr_data %>%
  group_by(CODE_GENDER,TARGET) %>%
  summarise(Count = n()) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(CODE_GENDER = reorder(CODE_GENDER,Count)) %>%
  mutate(TARGET = as.factor(TARGET)) %>%
  
  ggplot(aes(x = CODE_GENDER,y = Count,fill=TARGET)) +
  geom_bar(stat='identity', position=position_dodge(width=1),color=line_color) +
  scale_fill_manual("legend", values = color_combo )+
  geom_text(aes(x = CODE_GENDER, y = Count+2, label = paste0("",round(Count,0),sep=""),group=TARGET),
            position=position_dodge(width=1), size=text_size, colour = 'black',
            fontface = 'bold') +
  labs(x = 'Gender', 
       y = 'Percentage', 
       title = 'TARGET Count by Gender',
       subtitle = '0 - No Problem in Payment ; 1 - Problem in Payment')

#By Groups
cr_data %>% group_by(CODE_GENDER,TARGET)  %>% 
  summarise(Percentage=n())  %>% 
  group_by(CODE_GENDER) %>% 
  mutate(Percentage=Percentage/sum(Percentage)*100) %>% 
  
  ggplot(aes(x = CODE_GENDER,y = Percentage,fill=TARGET)) +
  geom_bar(stat='identity', position=position_dodge(width=1),color=line_color) +
  scale_fill_manual("legend", values = color_combo)+
  geom_text(aes(x = CODE_GENDER, y = Percentage+2, label = paste0("",round(Percentage,0), " % ",sep=""),group=TARGET),
            position=position_dodge(width=1), size=text_size, colour = 'black',
            fontface = 'bold') +
  labs(x = 'Gender', 
       y = 'Percentage', 
       title = 'Gender and Percentage',
       subtitle = '0 - No Problem in Payment ; 1 - Problem in Payment')

#Income Type Analysis=====================================================================================
cr_data %>% group_by(NAME_INCOME_TYPE,TARGET)  %>% 
  summarise(Percentage=n())  %>% 
  group_by(NAME_INCOME_TYPE) %>% 
  mutate(Percentage=Percentage/sum(Percentage)*100) %>% 
  
  ggplot(aes(x = NAME_INCOME_TYPE,y = Percentage,fill=TARGET)) +
  geom_bar(stat='identity', position=position_dodge(width=1),color=line_color) +
  scale_fill_manual("legend", values = color_combo)+
  geom_text(aes(x = NAME_INCOME_TYPE, y = Percentage+2, label = paste0("",round(Percentage,0), " % ",sep=""),group=TARGET),
            position=position_dodge(width=1), size=text_size, colour = 'black',
            fontface = 'bold') +
  labs(x = 'Income Type', 
       y = 'Percentage', 
       title = 'TARGET by Income Type',
       subtitle = '0 - No Problem in Payment ; 1 - Problem in Payment')

#Name Contract Type=====================================================================================
cr_data %>% group_by(NAME_CONTRACT_TYPE,TARGET)  %>% 
  summarise(Percentage=n())  %>% 
  group_by(NAME_CONTRACT_TYPE) %>% 
  mutate(Percentage=Percentage/sum(Percentage)*100) %>% 
  
  ggplot(aes(x = NAME_CONTRACT_TYPE,y = Percentage,fill=TARGET)) +
  geom_bar(stat='identity', position=position_dodge(width=1),color=line_color) +
  scale_fill_manual("legend", values = color_combo)+
  geom_text(aes(x = NAME_CONTRACT_TYPE, y = Percentage+2, label = paste0("",round(Percentage,0), " % ",sep=""),group=TARGET),
            position=position_dodge(width=1), size=text_size, colour = 'black',
            fontface = 'bold') +
  labs(x = 'Contract Type', 
       y = 'Percentage', 
       title = 'TARGET by Contract Type',
       subtitle = '0 - No Problem in Payment ; 1 - Problem in Payment')

#Owns Realty
cr_data %>% group_by(FLAG_OWN_REALTY ,TARGET)  %>% 
  summarise(Percentage=n())  %>% 
  group_by(FLAG_OWN_REALTY ) %>% 
  mutate(Percentage=Percentage/sum(Percentage)*100) %>% 
  
  ggplot(aes(x = FLAG_OWN_REALTY ,y = Percentage,fill=TARGET)) +
  geom_bar(stat='identity', position=position_dodge(width=1),color=line_color) +
  scale_fill_manual("legend", values = color_combo)+
  geom_text(aes(x = FLAG_OWN_REALTY , y = Percentage+2, label = paste0("",round(Percentage,0), " % ",sep=""),group=TARGET),
            position=position_dodge(width=1), size=text_size, colour = 'black',
            fontface = 'bold') +
  labs(x = 'Owns Real Estate', 
       y = 'Percentage', 
       title = 'TARGET by Owns Property',
       subtitle = '0 - No Problem in Payment ; 1 - Problem in Payment')

#Owns Car
cr_data %>% group_by(FLAG_OWN_CAR ,TARGET)  %>% 
  summarise(Percentage=n())  %>% 
  group_by(FLAG_OWN_CAR ) %>% 
  mutate(Percentage=Percentage/sum(Percentage)*100) %>% 
  
  ggplot(aes(x = FLAG_OWN_CAR ,y = Percentage,fill=TARGET)) +
  geom_bar(stat='identity', position=position_dodge(width=1),color=line_color) +
  scale_fill_manual("legend", values = color_combo)+
  geom_text(aes(x = FLAG_OWN_CAR , y = Percentage+2, label = paste0("",round(Percentage,0), " % ",sep=""),group=TARGET),
            position=position_dodge(width=1), size=text_size, colour = 'black',
            fontface = 'bold') +
  labs(x = 'Owns Car', 
       y = 'Percentage', 
       title = 'TARGET by Owns Car',
       subtitle = '0 - No Problem in Payment ; 1 - Problem in Payment')

#Name Family Status Analysis=====================================================================================
cr_data %>% group_by(NAME_FAMILY_STATUS,TARGET)  %>% 
  summarise(Percentage=n())  %>% 
  group_by(NAME_FAMILY_STATUS) %>% 
  mutate(Percentage=Percentage/sum(Percentage)*100) %>% 
  
  ggplot(aes(x = NAME_FAMILY_STATUS,y = Percentage,fill=TARGET)) +
  geom_bar(stat='identity', position=position_dodge(width=1),color=line_color) +
  scale_fill_manual("legend", values = color_combo)+
  geom_text(aes(x = NAME_FAMILY_STATUS, y = Percentage+2, label = paste0("",round(Percentage,0), " % ",sep=""),group=TARGET),
            position=position_dodge(width=1), size=text_size, colour = 'black',
            fontface = 'bold') +
  labs(x = 'Family Status', 
       y = 'Percentage', 
       title = 'TARGET by Family Status',
       subtitle = '0 - No Problem in Payment ; 1 - Problem in Payment')

#Accompanying person for the loan========================================================================
cr_data %>% group_by(NAME_TYPE_SUITE,TARGET)  %>% 
  summarise(Percentage=n())  %>% 
  group_by(NAME_TYPE_SUITE) %>% 
  mutate(Percentage=Percentage/sum(Percentage)*100) %>% 
  
  ggplot(aes(x = NAME_TYPE_SUITE,y = Percentage,fill=TARGET)) +
  geom_bar(stat='identity', position=position_dodge(width=1),color=line_color) +
  scale_fill_manual("legend", values = color_combo)+
  geom_text(aes(x = NAME_TYPE_SUITE, y = Percentage+2, label = paste0("",round(Percentage,0), " % ",sep=""),group=TARGET),
            position=position_dodge(width=1), size=text_size, colour = 'black',
            fontface = 'bold') +
  labs(x = 'Gender', 
       y = 'Percentage', 
       title = 'TARGET by Accompanying person for the loan',
       subtitle = '0 - No Problem in Payment ; 1 - Problem in Payment')

#Housing Type==========================================================================================
cr_data %>% group_by(NAME_HOUSING_TYPE,TARGET)  %>% 
  summarise(Percentage=n())  %>% 
  group_by(NAME_HOUSING_TYPE) %>% 
  mutate(Percentage=Percentage/sum(Percentage)*100) %>% 
  
  ggplot(aes(x = NAME_HOUSING_TYPE,y = Percentage,fill=TARGET)) +
  geom_bar(stat='identity', position=position_dodge(width=1),color=line_color) +
  scale_fill_manual("legend", values = color_combo)+
  geom_text(aes(x = NAME_HOUSING_TYPE, y = Percentage+2, label = paste0("",round(Percentage,0), " % ",sep=""),group=TARGET),
            position=position_dodge(width=1), size=text_size, colour = 'black',
            fontface = 'bold') +
  labs(x = 'Housing Type', 
       y = 'Percentage', 
       title = 'TARGET by Housing Type',
       subtitle = '0 - No Problem in Payment ; 1 - Problem in Payment')

#Count of Family Members=================================================================================
cr_data %>% group_by(CNT_FAM_MEMBERS,TARGET)  %>% 
  summarise(Percentage=n())  %>% 
  group_by(CNT_FAM_MEMBERS) %>% 
  mutate(Percentage=Percentage/sum(Percentage)*100) %>% 
  
  ggplot(aes(x = CNT_FAM_MEMBERS,y = Percentage,fill=TARGET)) +
  geom_bar(stat='identity', position=position_dodge(width=1),color=line_color) +
  scale_fill_manual("legend", values = color_combo)+
  geom_text(aes(x = CNT_FAM_MEMBERS, y = Percentage+2, label = paste0("",round(Percentage,0), " % ",sep=""),group=TARGET),
            position=position_dodge(width=1), size=text_size, colour = 'black',
            fontface = 'bold') +
  labs(x = 'Housing Type', 
       y = 'Percentage', 
       title = 'TARGET by Housing Type',
       subtitle = '0 - No Problem in Payment ; 1 - Problem in Payment')

#Organization Type=====================================================================================
ORG_Plot1 <- cr_data %>% 
  group_by(ORGANIZATION_TYPE,TARGET)  %>% 
  summarise(Percentage=n())  %>% 
  group_by(ORGANIZATION_TYPE) %>% 
  mutate(Percentage=Percentage/sum(Percentage)*100) %>%
  filter((Percentage>50 & Percentage <89)|(Percentage>11 & Percentage <50)) %>% 
  
  ggplot(aes(x = ORGANIZATION_TYPE,y = Percentage,fill=TARGET)) +
  geom_bar(stat='identity', position=position_dodge(width=1),color=line_color) +
  scale_fill_manual("legend", values = color_combo)+
  geom_text(aes(x = ORGANIZATION_TYPE, y = Percentage+2, label = paste0("",round(Percentage,0), " % ",sep=""),group=TARGET),
            position=position_dodge(width=1), size=text_size, colour = 'black',
            fontface = 'bold') +
  labs(x = 'Organization Type', 
       y = 'Percentage', 
       title = 'High Defaulter Rate',
       subtitle = '0 - No Problem in Payment ; 1 - Problem in Payment')

ORG_Plot2 <- cr_data %>% 
  group_by(ORGANIZATION_TYPE,TARGET)  %>% 
  summarise(Percentage=n())  %>% 
  group_by(ORGANIZATION_TYPE) %>% 
  mutate(Percentage=Percentage/sum(Percentage)*100) %>%
  filter((Percentage>95)|(Percentage<5)) %>% 
  
  ggplot(aes(x = ORGANIZATION_TYPE,y = Percentage,fill=TARGET)) +
  geom_bar(stat='identity', position=position_dodge(width=1),color=line_color) +
  scale_fill_manual("legend", values = color_combo)+
  geom_text(aes(x = ORGANIZATION_TYPE, y = Percentage+2, label = paste0("",round(Percentage,0), " % ",sep=""),group=TARGET),
            position=position_dodge(width=1), size=text_size, colour = 'black',
            fontface = 'bold') +
  labs(x = 'Organization Type', 
       y = 'Percentage', 
       title = 'Low Defaulter Rate',
       subtitle = '0 - No Problem in Payment ; 1 - Problem in Payment')

p <-plot_grid(ORG_Plot1, ORG_Plot2, nrow = 2)
title <- ggdraw() + draw_label("Defaulter Rate by Organization Type", fontface='bold')
plot_grid(title, p, ncol=1 , rel_heights=c(0.1, 1))

#Distribution of Credit Amount of loan==============================================================================
library(cowplot) 

CA_Plot1 <- cr_data %>% 
  filter(AMT_CREDIT < 2e6) %>%
  ggplot(aes(x = AMT_CREDIT)) +
  geom_histogram(bins = 30, fill = single_color,color = line_color) +
  labs(x= 'Amount Credit',y = 'Count')#, title = paste("Overall Distribution of", ' Amount Credit ')) 

CA_Plot2 <- cr_data %>% 
  filter(TARGET ==0) %>% 
  filter(AMT_CREDIT < 2e6) %>%
  ggplot(aes(x = AMT_CREDIT)) +
  geom_histogram(bins = 20, fill = single_color,color = line_color) +
  labs(x= 'Amount Credit',y = 'Count')#, title = paste("Distribution of", ' Amount Credit ')) 

CA_Plot3 <- cr_data %>% 
  filter(TARGET ==1) %>% 
  filter(AMT_CREDIT < 2e6) %>%
  ggplot(aes(x = AMT_CREDIT)) +
  geom_histogram(bins = 20, fill = single_color,color = line_color) +
  labs(x= 'Amount Credit',y = 'Count')#, title = paste("Distribution of", ' Amount Credit '))  

p <-plot_grid(CA_Plot1,CA_Plot2, CA_Plot3,nrow=1,ncol=3, labels = c("OVERALL","TARGET = 0","TARGET = 1"))
title <- ggdraw() + draw_label("Distribution of Amount Credit by TARGET", fontface='bold')
plot_grid(title, p, ncol=1, rel_heights=c(0.1, 1))

#Distribution of Income Amount of loan==============================================================================
library(cowplot) 

CA_Plot1 <- cr_data %>% 
  filter(AMT_INCOME_TOTAL < 5e5) %>%
  ggplot(aes(x = AMT_INCOME_TOTAL)) +
  geom_histogram(bins = 30, fill = single_color,color = line_color) +
  labs(x= 'Amount Income Total',y = 'Count')#, title = paste("Overall Distribution of", ' Amount Income Total')) 

CA_Plot2 <- cr_data %>% 
  filter(TARGET ==0) %>% 
  filter(AMT_INCOME_TOTAL < 5e5) %>%
  ggplot(aes(x = AMT_INCOME_TOTAL)) +
  geom_histogram(bins = 20, fill = single_color,color = line_color) +
  labs(x= 'Amount Income Total',y = 'Count')#, title = paste("Distribution of", ' Amount Income Total')) 

CA_Plot3 <- cr_data %>% 
  filter(TARGET ==1) %>% 
  filter(AMT_INCOME_TOTAL < 5e5) %>%
  ggplot(aes(x = AMT_INCOME_TOTAL)) +
  geom_histogram(bins = 20, fill = single_color,color = line_color) +
  labs(x= 'Amount Income Total',y = 'Count')#, title = paste("Distribution of", ' Amount Income Total '))  

p <-plot_grid(CA_Plot1,CA_Plot2, CA_Plot3,nrow=1,ncol=3, labels = c("OVERALL","TARGET = 0","TARGET = 1"))
title <- ggdraw() + draw_label("Distribution of Amount Income Total by TARGET", fontface='bold')
plot_grid(title, p, ncol=1, rel_heights=c(0.1, 1))

#Distribution of Annuity Amount of Loan==================================================================
CA_Plot1 <- cr_data %>% 
  filter(AMT_ANNUITY    < 2e5) %>%
  ggplot(aes(x = AMT_ANNUITY   )) +
  geom_histogram(bins = 30, fill = single_color,color = line_color) +
  labs(x= 'Amount Annuity',y = 'Count')#, title = paste("Overall Distribution of", ' Amount Annuity ')) 

CA_Plot2 <- cr_data %>% 
  filter(TARGET ==0) %>% 
  filter(AMT_ANNUITY    < 2e5) %>%
  ggplot(aes(x = AMT_ANNUITY   )) +
  geom_histogram(bins = 20, fill = single_color,color = line_color) +
  labs(x= 'Amount Annuity',y = 'Count')#, title = paste("Distribution of", ' Amount Annuity ')) 

CA_Plot3 <- cr_data %>% 
  filter(TARGET ==1) %>% 
  filter(AMT_ANNUITY    < 2e5) %>%
  ggplot(aes(x = AMT_ANNUITY   )) +
  geom_histogram(bins = 20, fill = single_color,color = line_color) +
  labs(x= 'Amount Annuity',y = 'Count')#, title = paste("Distribution of", ' Amount Annuity '))  

p <-plot_grid(CA_Plot1,CA_Plot2, CA_Plot3,nrow=1,ncol=3, labels = c("OVERALL","TARGET = 0","TARGET = 1"))
title <- ggdraw() + draw_label("Distribution of Amount Annuity by TARGET", fontface='bold')
plot_grid(title, p, ncol=1, rel_heights=c(0.1, 1))

#Days Birth============================================================================================
CA_Plot1 <- cr_data %>% 
  filter(DAYS_BIRTH    < 2e5) %>%
  ggplot(aes(x = DAYS_BIRTH   )) +
  geom_histogram(bins = 30, fill = single_color,color = line_color) +
  labs(x= 'Amount Annuity',y = 'Count')#, title = paste("Overall Distribution of", ' Amount Annuity ')) 

CA_Plot2 <- cr_data %>% 
  filter(TARGET ==0) %>% 
  filter(DAYS_BIRTH    < 2e5) %>%
  ggplot(aes(x = DAYS_BIRTH   )) +
  geom_histogram(bins = 20, fill = single_color,color = line_color) +
  labs(x= 'Amount Annuity',y = 'Count')#, title = paste("Distribution of", ' Amount Annuity ')) 

CA_Plot3 <- cr_data %>% 
  filter(TARGET ==1) %>% 
  filter(DAYS_BIRTH    < 2e5) %>%
  ggplot(aes(x = DAYS_BIRTH   )) +
  geom_histogram(bins = 20, fill = single_color,color = line_color) +
  labs(x= 'Amount Annuity',y = 'Count')#, title = paste("Distribution of", ' Amount Annuity '))  

p <-plot_grid(CA_Plot1,CA_Plot2, CA_Plot3,nrow=1,ncol=3, labels = c("OVERALL","TARGET = 0","TARGET = 1"))
title <- ggdraw() + draw_label("Distribution of Amount Annuity by TARGET", fontface='bold')
plot_grid(title, p, ncol=1, rel_heights=c(0.1, 1))

#Days Employed
CA_Plot1 <- cr_data %>% 
  filter(DAYS_EMPLOYED    < 2e5) %>%
  ggplot(aes(x = DAYS_EMPLOYED   )) +
  geom_histogram(bins = 30, fill = single_color,color = line_color) +
  labs(x= 'Amount Annuity',y = 'Count')#, title = paste("Overall Distribution of", ' Amount Annuity ')) 

CA_Plot2 <- cr_data %>% 
  filter(TARGET ==0) %>% 
  filter(DAYS_EMPLOYED    < 2e5) %>%
  ggplot(aes(x = DAYS_EMPLOYED   )) +
  geom_histogram(bins = 20, fill = single_color,color = line_color) +
  labs(x= 'Amount Annuity',y = 'Count')#, title = paste("Distribution of", ' Amount Annuity ')) 

CA_Plot3 <- cr_data %>% 
  filter(TARGET ==1) %>% 
  filter(DAYS_EMPLOYED    < 2e5) %>%
  ggplot(aes(x = DAYS_EMPLOYED   )) +
  geom_histogram(bins = 20, fill = single_color,color = line_color) +
  labs(x= 'Amount Annuity',y = 'Count')#, title = paste("Distribution of", ' Amount Annuity '))  

p <-plot_grid(CA_Plot1,CA_Plot2, CA_Plot3,nrow=1,ncol=3, labels = c("OVERALL","TARGET = 0","TARGET = 1"))
title <- ggdraw() + draw_label("Distribution of Amount Annuity by TARGET", fontface='bold')
plot_grid(title, p, ncol=1, rel_heights=c(0.1, 1))

#EXT SOURCE ==============================================================================
gc()
EXT_Plot1 <- ggplot(cr_data , aes(x=as.factor(TARGET),y=EXT_SOURCE_2)) + 
  geom_boxplot(fill=single_color, width = 0.4,)+
  labs(x= 'TARGET',y = "EXT_SOURCE_2")+ coord_flip()

EXT_Plot2 <- cr_data %>% 
  filter(TARGET ==0) %>% 
  ggplot(aes(x = EXT_SOURCE_2)) +
  geom_histogram(bins = 100, fill = single_color,color = line_color) +
  labs(x= 'EXT_SOURCE_2',y = 'Count')

EXT_Plot3 <- cr_data %>% 
  filter(TARGET ==1) %>% 
  ggplot(aes(x = EXT_SOURCE_2)) +
  geom_histogram(bins = 100, fill = single_color,color = line_color) +
  labs(x= 'EXT_SOURCE_2',y = 'Count')

EXT_Plot4 <- ggplot(cr_data , aes(x=as.factor(TARGET),y=EXT_SOURCE_3)) + 
  geom_boxplot(fill=single_color, width = 0.4,)+
  labs(x= 'TARGET',y = "EXT_SOURCE_3")+ coord_flip()

EXT_Plot5 <- cr_data %>% 
  filter(TARGET ==0) %>% 
  ggplot(aes(x = EXT_SOURCE_3)) +
  geom_histogram(bins = 50, fill = single_color,color = line_color) +
  labs(x= 'EXT_SOURCE_3',y = 'Count')

EXT_Plot6 <- cr_data %>% 
  filter(TARGET ==1) %>%
  ggplot(aes(x = EXT_SOURCE_3)) +
  geom_histogram(bins = 50, fill = single_color,color = line_color) +
  labs(x= 'EXT_SOURCE_3',y = 'Count')
gc()
p <-plot_grid(EXT_Plot1, EXT_Plot2,EXT_Plot3, EXT_Plot4, EXT_Plot5, EXT_Plot6,
              nrow=2, ncol = 3, labels = c(" Box Plot","TARGET = 0","TARGET = 1"," Box Plot","TARGET = 0","TARGET = 1"))
title <- ggdraw() + draw_label("Distribution of EXT_SOURCE by TARGET", fontface='bold')
plot_grid(title, p, ncol=1, rel_heights=c(0.1, 1))

#Corr Plot============
Imp_Cols <-  l %>% pull(var)
cr_selected <- cr_train[c(Imp_Cols[0:30],'TARGET')]
cr_selected <-  cr_selected %>% select_if(is.numeric)

cormat <- round(cor(cr_selected),2)

library(reshape2)
melted_cormat <- melt(cormat)
head(melted_cormat)

ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()+
  scale_fill_gradient2(low = "white", high = single_color, mid = "darkgoldenrod1", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") 

#==========================================================================================================
#Logistic Regression
#==========================================================================================================
library(e1071)       #for calculating variable importance
library(caret)       #for general model fitting
library(rpart)       #for fitting decision trees
library(ipred)       #for fitting bagged decision trees
library(dplyr)       #for data wrangling
library(tidyverse)
library(aod)
library(ggplot2)
library(caTools)

cr_data2 <- cr_data[2:72]

random_sample <- createDataPartition(cr_data2$TARGET,
                                     p = 0.8, list = FALSE)
cr_data2 <- as.data.frame(cr_data2)

train <- cr_data2[random_sample,]
test <- cr_data2[-random_sample,]

trctrl <- trainControl(method = "cv", number = 10, savePredictions=TRUE)
model <- train(TARGET ~.,data=train, trControl = trctrl, method = "glm", family=binomial(link='logit'))
print(model)
summary(model)
vi <- varImp(model)
plot(varImp(model),col = "red3", top=20)

#predict the outcome
predictions <- predict(model, newdata=test)

#create confision matrix
confusionMatrix(predictions, test$TARGET)

#run test data
res_train <- predict(model, train, type="response")
res_test <- predict(model, test, type="response")
hist(res_test)
res_test <- ifelse(res_test <= 0.1, 0, 1)
res_test <- data.frame(res_test)

#Validate the model - confusion matrix
confmatrix <- table(Actual_Value=test$TARGET, Predicted_Value = res_test >0.5)
confmatrix

#sensitivity and specificity
library(caret)
sensitivity(as.factor(test$TARGET), as.factor(res_test$res_test))
specificity(as.factor(test$TARGET), as.factor(res_test$res_test))

#accuracy
accuracy <- (confmatrix[[1,1]] + confmatrix[[2,2]])/sum(confmatrix)
accuracy

#==========================================================================================================
#kNN
#==========================================================================================================
cr_temp <- cr_data %>% select_if(is.numeric)
#for(i in 1:ncol(cr_temp)){
#  cr_temp[,i] = scale(cr_temp[,i])
#}
cr_temp = as.data.frame(scale(cr_temp))
cr_temp = cr_temp %>% select(!FLAG_MOBIL)
cr_temp['TARGET'] <- cr_data['TARGET']

  #cr_temp <-  cr_temp[1:46000,]
train_ind <- sample(1:nrow(cr_temp), round(nrow(cr_temp)*2/3))
cr_train <- cr_temp[train_ind,]
cr_val <- cr_temp[-train_ind, ]
cr_train_tr <- cr_temp[train_ind,'TARGET']
cr_val_tr <- cr_temp[-train_ind,'TARGET']

cr_train <- cr_train %>% select(!c(TARGET))
cr_val <- cr_val %>% select(!c(TARGET))

near = knn(train=cr_train,test=cr_val,cl=cr_train_tr,k=5)

#add back the numeric variables that we once converted to factor variables
cr_temp1 = cr_data[num_2_factor_names]%>%select_if(is.factor)
for(i in 1:20){
  cr_temp1[[i]] = as.numeric(levels(cr_temp1[[i]]))[cr_temp1[[i]]]
}
cr_temp2 = cbind(cr_temp,cr_temp1)
  #cr_temp2 <-  cr_temp2[1:46000,]
train_ind <- sample(1:nrow(cr_temp2), round(nrow(cr_temp2)*2/3))
cr_train <- cr_temp2[train_ind,]
cr_val <- cr_temp2[-train_ind, ]
cr_train_tr <- cr_temp2[train_ind,'TARGET']
cr_val_tr <- cr_temp2[-train_ind,'TARGET']

near1 = knn(train=cr_train,test=cr_val,cl=cr_train_tr,k=20)

#Validate the model - confusion matrix, Accuracy
tab = table(near,cr_val_tr)
accuracy <- function(x){sum(diag(x)/sum(rowSums(x)))}
accuracy(tab)

##K-Fold Cross Validation##
n=dim(cr_data)[1]
kcv = 10
n0 = round(n/kcv,0)
out_MSE = matrix(0,kcv,100)

used = NULL
set = 1:n

for(j in 1:kcv){
  
  if(n0<length(set)){val = sample(set,n0)}
  if(n0>=length(set)){val=set}
  
  train_i = cr_temp2[-val,]
  test_i = cr_temp2[val,]
  
  for(i in 1:100){
    near = kknn(TARGET~.,train_i,test_i,k=i,kernel = "rectangular")
    aux = mean((test_i[,61]-near$fitted)^2)
    out_MSE[j,i] = aux
  }
  used = union(used,val)
  set = (1:n)[-used]
  
  cat(j,'\n')
}

mMSE = apply(out_MSE,2,mean)

plot(log(1/(1:100)),sqrt(mMSE),xlab="Complexity (log(1/k))",ylab="out-of-sample RMSE",col=4,lwd=2,type="l",cex.lab=1.2,main=paste("kfold(",kcv,")"))
best = which.min(mMSE)
text(log(1/best),sqrt(mMSE[best])+0.1,paste("k=",best),col=2,cex=1.2)
text(log(1/2),sqrt(mMSE[2])+0.3,paste("k=",2),col=2,cex=1.2)
text(log(1/100)+0.4,sqrt(mMSE[100]),paste("k=",100),col=2,cex=1.2)

#==========================================================================================================
#Random Forrest
#==========================================================================================================
#train, validate split
d_train = sample(1:nrow(cr_data),round(nrow(cr_data)*2/3))
cr_train = cr_data[d_train,]
cr_vali = cr_data[-d_train,]

#Modeling
library(randomForest)
bag_cr = randomForest(TARGET~.-ORGANIZATION_TYPE,data = cr_train, mtry = 8, ntree = 300, importance = TRUE)
varImpPlot(bag_cr)#identify important variables

result = predict(bag_cr,cr_vali)
confusion_matrix = table(cr_vali$TARGET,result)

#use undersampling to balance data
library(ROSE)#oversampling, undersampling
balance_train = ovun.sample(TARGET~., data = cr_train, method = 'both')$data
table(balance_train$TARGET)

bag_cr1 = randomForest(TARGET~EXT_SOURCE_2+EXT_SOURCE_3+DAYS_EMPLOYED+
                         DAYS_REGISTRATION+DAYS_ID_PUBLISH+DAYS_BIRTH+
                         AMT_GOODS_PRICE+AMT_CREDIT+AMT_INCOME_TOTAL+AMT_ANNUITY+OBS_60_CNT_SOCIAL_CIRCLE+
                         OBS_30_CNT_SOCIAL_CIRCLE+DAYS_LAST_PHONE_CHANGE+WEEKDAY_APPR_PROCESS_START,
                         data= balance_train, mtry = 4, ntree = 300, importance = TRUE)
result = predict(bag_cr1,balance_train)

#Validate the model - confusion matrix, f score
confusion_matrix = table(balance_train$TARGET,result)
f_score = function(matrix){
  (2*(matrix[2,2]/(matrix[2,1]+matrix[2,2]))*(matrix[2,2]/(matrix[1,2]+matrix[2,2])))/(matrix[2,2]/(matrix[2,1]+matrix[2,2]))+(matrix[2,2]/(matrix[1,2]+matrix[2,2]))
}#F-Measure = (2 * Precision * Recall) / (Precision + Recall) function
f_score(confusion_matrix)

#==========================================================================================================
#Boosting
#==========================================================================================================
#Train test split========================================================================================
set.seed(1)

#Drop ID Column
cr_data <- cr_data %>% select(!SK_ID_CURR)  

#Sample
train_ind <- sample(1:nrow(cr_data), round(nrow(cr_data)*2/3))

#Train & Test Split
cr_train <- cr_data[train_ind,]
cr_val <- cr_data[-train_ind, ]

#Get Target Distribution
cr_train %>% group_by(TARGET) %>% summarise(Count = n()/nrow(cr_train)*100)
cr_val %>% group_by(TARGET) %>% summarise(Count = n()/nrow(cr_val)*100)

#Gradient Boosting model=================================================================================
boosting_model = gbm(TARGET~.,data=cr_train, distribution= "bernoulli", n.trees=500,shrinkage=0.01)

summary(boosting_model)

#boosting_model = train(TARGET~., data=cr_train,method="gbm", distribution ="bernoulli", n.trees=500,shrinkage=0.01)
#plot(varImp(boosting_model),top=20) 

#Training Scores=========================================================================================
#Predict Using Boosting
pred_boosting <- predict(boosting_model,cr_train,type='response')
pred_boosting <- ifelse(pred_boosting>=0.5,1,0)

#F1- Score
F1_Score(y_true = cr_train$TARGET, y_pred = pred_boosting)

#Check AUC
AUC(cr_train$TARGET, pred_boosting)

#Observe Confusion Matrix
confusionMatrix(as.factor(pred_boosting), as.factor(cr_train$TARGET), positive = '1')

#Validation Scores=====================================================================================
#Predict Using Boosting
pred_boosting <- predict(boosting_model,cr_val,type='response')
pred_boosting <- ifelse(pred_boosting>=0.5,1,0)

#F1- Score
F1_Score(cr_val$TARGET, pred_boosting, positive = '1')

#Check AUC
AUC(cr_val$TARGET, pred_boosting)

#Observe Confusion Matrix
confusionMatrix(as.factor(pred_boosting), as.factor(cr_val$TARGET), positive = '1')

# Sensitivity is extremely low, we can clearly mention that one of the classes is dominated 
# over another class. Suppose if you're interested in zero class then this model is quite a good 
# one and if you want to predict class 1 then need to improvise the model.

#Oversamping==========================================================================================
cr_over <- ovun.sample(TARGET~., data = cr_train, method = "both")$data
table(cr_over$TARGET)

boosting_model = gbm(TARGET~.,data=cr_over, distribution= "bernoulli", n.trees=100,shrinkage=0.01)

l = summary(boosting_model,plotit=F)

#Training Scores=======================================================================================
#Predict Using Boosting
pred_boosting <- predict(boosting_model,cr_train,type='response')
pred_boosting <- ifelse(pred_boosting>=0.5,1,0)

#F1- Score
F1_Score(cr_train$TARGET, pred_boosting, positive = NULL)

#Check AUC
AUC(cr_train$TARGET, pred_boosting)

#Observe Confusion Matrix
confusionMatrix(as.factor(pred_boosting), as.factor(cr_train$TARGET), positive = '1')

#Validation Scores=====================================================================================
#Predict Using Boosting
pred_boosting <- predict(boosting_model,cr_val,type='response')
pred_boosting <- ifelse(pred_boosting>=0.5,1,0)

#F1- Score
F1_Score(cr_val$TARGET, pred_boosting, positive = NULL)

#Check AUC
AUC(cr_val$TARGET, pred_boosting)

#Observe Confusion Matrix
confusionMatrix(as.factor(pred_boosting), as.factor(cr_val$TARGET), positive = '1')

#Undersamping==========================================================================================
cr_under <- ovun.sample(TARGET~., data = cr_train, method = "under")$data
table(cr_under$TARGET)
boosting_model = gbm(TARGET~.,data=cr_under, distribution= "bernoulli", n.trees=100,shrinkage=0.01)

summary(boosting_model)

#Training Scores=======================================================================================
#Predict Using Boosting
pred_boosting <- predict(boosting_model,cr_under,type='response')
pred_boosting <- ifelse(pred_boosting>=0.5,1,0)

#F1- Score
F1_Score(cr_under$TARGET, pred_boosting, positive = NULL)

#Check AUC
AUC(cr_under$TARGET, pred_boosting)

#Observe Confusion Matrix
confusionMatrix(as.factor(pred_boosting), as.factor(cr_under$TARGET), positive = '1')

#Validation Scores=====================================================================================
pred_boosting <- predict(boosting_model,cr_val,type='response')
pred_boosting <- ifelse(pred_boosting>=0.5,1,0)

#F1- Score
F1_Score(cr_val$TARGET, pred_boosting, positive = NULL)

#Check AUC
AUC(cr_val$TARGET, pred_boosting)

#Observe Confusion Matrix
confusionMatrix(as.factor(pred_boosting), as.factor(cr_val$TARGET), positive = '1')

#Combined==========================================================================================
cr_both <- ovun.sample(TARGET~., data = cr_train, method = "both")$data
table(cr_both$TARGET)
boosting_model = gbm(TARGET~.,data=cr_both, distribution= "bernoulli", n.trees=100,shrinkage=0.01)
summary(boosting_model,plotit=F)

#Training Scores=======================================================================================
#Predict Using Boosting
pred_boosting <- predict(boosting_model,cr_both,type='response')
pred_boosting <- ifelse(pred_boosting>=0.5,1,0)

#F1- Score
F1_Score(cr_both$TARGET, pred_boosting, positive = NULL)

#Check AUC
AUC(cr_both$TARGET, pred_boosting)

#Observe Confusion Matrix
confusionMatrix(as.factor(pred_boosting), as.factor(cr_both$TARGET), positive = '1')

#Validation Scores=====================================================================================
pred_boosting <- predict(boosting_model,cr_both,type='response')
pred_boosting <- ifelse(pred_boosting>=0.5,1,0)

#F1- Score
F1_Score(cr_both$TARGET, pred_boosting, positive = NULL)

#Check AUC
AUC(cr_both$TARGET, pred_boosting)

#Observe Confusion Matrix
confusionMatrix(as.factor(pred_boosting), as.factor(cr_both$TARGET), positive = '1')

#Trying different tree size=============================================================================
sequence=c(100,500,1000,2500,5000)
for (n in sequence)
{
  cr_under <- ovun.sample(TARGET~., data = cr_train, method = "under")$data
  table(cr_under$TARGET)
  boosting_model = gbm(TARGET~.,data=cr_under, distribution= "bernoulli", n.trees=2500,shrinkage=0.01,cv.folds=5)
  
  pred_boosting <- predict(boosting_model,cr_both,type='response')
  pred_boosting <- ifelse(pred_boosting>=0.5,1,0)
  
  print("Tree: ")
  print(confusionMatrix(as.factor(pred_boosting), as.factor(cr_both$TARGET), positive = '1'))
}

#With Selected Columns======================================================================================================
set.seed(1)

#Choose top 30 columns
Imp_Cols <-  l %>% pull(var)
cr_selected <- cr_train[c(Imp_Cols[0:30],'TARGET')]

cr_both <- ovun.sample(TARGET~., data = cr_selected, method = "under")$data
table(cr_both$TARGET)

boosting_model = gbm(TARGET~.,data=cr_both, distribution= "bernoulli", n.trees=1000,shrinkage=0.01)

summary(boosting_model,plotit=F)

#Training Scores=======================================================================================
pred_boosting <- predict(boosting_model,cr_both,type='response')
pred_boosting <- ifelse(pred_boosting>=0.5,1,0)

F1_Score(cr_both$TARGET, pred_boosting, positive = NULL) #F1- Score
AUC(cr_both$TARGET, pred_boosting) #AUC
confusionMatrix(as.factor(pred_boosting), as.factor(cr_both$TARGET), positive = '1') #Confusion Matrix

#Validation Scores=====================================================================================
pred_boosting <- predict(boosting_model,cr_val,type='response')
pred_boosting <- ifelse(pred_boosting>=0.5,1,0)

F1_Score(cr_val$TARGET, pred_boosting, positive = NULL) #F1- Score
AUC(cr_val$TARGET, pred_boosting) #AUC
confusionMatrix(as.factor(pred_boosting), as.factor(cr_val$TARGET), positive = '1') #Confusion Matrix


