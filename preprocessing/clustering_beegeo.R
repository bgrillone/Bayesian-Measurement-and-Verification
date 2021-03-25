## GROUP 1 - CLUSTERING -------

# Load national holidays  
national_days <- fread("https://analisi.transparenciacatalunya.cat/api/views/yf2b-mjr6/rows.csv?accessType=DOWNLOAD&sorting=true",
                       data.table=F) # National holidays for Catalonia (Spain)
national_days$Data <- as.Date(national_days$Data, format="%d/%m/%Y")

list_of_holidays <- function(national_days, days_list){
  holidays <- sort(unique(c(national_days$Data,
                            as.Date(c(mapply(function(d){seq(d-7,d, by="days")},national_days[national_days$`Nom del festiu`=="Dilluns de Pasqua Florida","Data"]))),
                            as.Date(c(mapply(function(d){seq(d-11,d, by="days")},national_days[national_days$`Nom del festiu`=="Reis","Data"]))),
                            as.Date(days_list)[month(days_list)==8],
                            days_list[strftime(days_list,"%u") %in% c("6","7") & !(month(days_list) %in% c(5,6,9,10))]
  )))
  holidays <- holidays[holidays %in% unique(days_list)]
  bridges <- holidays[holidays - lag(holidays,1,holidays[1]-1) - 1 == 1]
  if(length(bridges)>0){
    holidays <- c(holidays,bridges-1)
  }
  return(holidays)
}

normalize_range_int <- function(df,inf,sup,specs=NULL,threshold_for_min=NULL){
  
  if(sup>inf){
    
    if(!is.null(ncol(df))){
      if (is.null(specs)){
        specs <- mapply(function(i){c("min"=min(df[,i],na.rm=T),
                                      "max"=max(df[,i],na.rm=T))},1:ncol(df))
      }
      result <- list(
        "norm" = mapply(
          function(i){
            x <- df[,i]
            if (specs["min",i]==specs["max",i]){
              rep(inf,length(x))
            } else {
              r <- ((x-specs["min",i])/(specs["max",i]-specs["min",i]))*(sup-inf)+inf
              if(!is.null(threshold_for_min)){
                r <- ifelse(x>=threshold_for_min,r,inf)
              }
              return(r)
            }
          },
          1:ncol(df)
        ),
        "specs" = specs
      )
      if(!is.matrix(result$norm)){
        result$norm <- t(result$norm)
        colnames(result$norm) <- colnames(df)
      }
      return(result)
    } else{
      x <- df
      if (is.null(specs)){
        specs <- c("min" = min(x,na.rm=T), "max" = max(x,na.rm=T))
      }
      if (specs["min"]==specs["max"]){
        rep(inf,length(x))
      } else {
        r <- ((x-specs["min"])/(specs["max"]-specs["min"]))*(sup-inf)+inf
        if(!is.null(threshold_for_min)){
          r <- ifelse(x>=threshold_for_min,r,inf)
        }
      }
      return(r)
    }
  }
}

# normalize_range_int_column <- function(x,inf,sup,threshold_for_min=NULL){
#   
#   if(sup>inf){
#     if (max(x)==min(x)){
#       rep(inf,length(x))
#     } else {
#       r <- ((x-min(x))/(max(x)-min(x)))*(sup-inf)+inf
#       if(!is.null(threshold_for_min)){
#         r <- ifelse(x>=threshold_for_min,r,inf)
#       }
#       return(r)
#     }
#   }
# }
# 


normalize_perc_cons_day <- function(df,cores=8){
  
  return(
    as.data.frame(
      t(
        if(cores>1){
          mcmapply(
            function(i){as.numeric(df[i,]/(mean(as.numeric(df[i,]),na.rm=T)*24))},
            1:nrow(df),
            mc.cores = cores
          )
        }else{
          mapply(
            function(i){as.numeric(df[i,]/(mean(as.numeric(df[i,]),na.rm=T)*24))},
            1:nrow(df)
          )
        }
      )
    )
  )
}

normalize_znorm <- function(df, spec_med_sd=NULL){
  if (is.null(spec_med_sd)){
    spec_med_sd <- mapply(function(i){c("median"=median(df[,i],na.rm=T),
                                        "sd"=sd(df[,i],na.rm=T))},1:ncol(df))
  }
  
  return(
    list(
      "norm"=as.data.frame(
        mapply(
          function(i){(as.numeric(df[,i])-spec_med_sd["median",i])/spec_med_sd["sd",i]},
          1:ncol(df)
        )
      ),
      "specs"=spec_med_sd
    )
  )
}

smartAgg <- function(df, by, ..., catN=T, printAgg=F) {
  args <- list(...)
  dt <- as.data.table(df)
  
  ## organizing agg Methods and variable names into 2 separate lists
  aggMethod <- list()
  vars <- list()
  j<-1
  for(i in seq(1,length(args),2)) {
    aggMethod[[j]] <- args[[i]]
    vars[[j]] <- args[[i+1]]
    if(class(vars[[j]]) %in% c('integer', 'numeric')) vars[[j]] <- names(df)[vars[[j]]]
    j<-j+1
  }
  
  ## creat line to exec
  k<-0
  varL <- vector()
  for(j in 1:length(aggMethod)){
    for(i in 1:length(vars[[j]])){
      if(vars[[j]][i] %in% names(df)){
        if(class(aggMethod[[j]])=='function') {
          afun <- paste0('af',j)
          assign(afun, aggMethod[[j]])
          laf2 <- as.list(formals(get(afun)))
          laf2[which(lapply(laf2, nchar)==0)] <- vars[[j]][i] 
          rhstmp <- paste(unlist(lapply(seq_along(laf2), function(y,n,i) paste0(n[[i]], '=', y[[i]]), n=names(laf2), y=laf2)), collapse=',')
          tmp <- paste(vars[[j]][i], '=', afun, '(', rhstmp, ')', sep='') # anonymous functions
        } else {         
          tmp <- paste(vars[[j]][i], '=', aggMethod[[j]], '(', vars[[j]][i], ')', sep='') #non-anonymous functions
        }
        k <- k+1
        varL[k] <- tmp
      } else {print(paste('WARNING: ', vars[[j]][i], ' not in dataframe', sep=''))}
    }
  }
  varL <- paste(varL, collapse=', ')
  if(catN==T) varL <- paste(varL, ',countPerBin=length(', vars[[1]][1], ')', sep='')  
  
  ## actually creating aggregation command and executing it
  line2exec <- paste('dtAgg <- dt[,list(', varL, '), by=list(', paste(by,collapse=','), ')]', sep='')
  if(printAgg==T) print(line2exec)
  eval(parse(text=line2exec))
  dfAgg <- data.frame(dtAgg)
  
  return(dfAgg)
}


norm_load_curves <- function(df,tz_local,time_column="t",value_column="v",temperature_column="outdoor_temp", perc_cons=T, n_dayparts=24, 
                             norm_specs=NULL, input_vars = c("load_curves","days_weekend","days_of_the_week", "daily_cons", "daily_temp"),
                             filter_na=T){
  
  df_agg <- as.data.frame(df)
  df_agg <- data.frame(
    "time"= df_agg[,time_column],
    "temperature"= df_agg[,temperature_column],
    "day"= as.Date(df_agg[,time_column], tz=tz_local),
    "value"=df_agg[,value_column],
    "dayhour"=sprintf("%02i:00",hour(with_tz(df_agg[,time_column], tz=tz_local))),
    "daypart"=ceiling(hour(with_tz(df_agg[,time_column], tz=tz_local))/(24/n_dayparts)),
    stringsAsFactors = F
  )
  df_agg_d <- smartAgg(df_agg,"day",function(x){mean(x,na.rm=T)*24},"value",function(x){mean(x,na.rm=T)},"temperature",catN = F)
  df_agg <- df_agg[!duplicated(df_agg[,c("dayhour","day")]),]
  
  if(n_dayparts==24){
    df_agg_to_spread <- df_agg[,c("day","value","dayhour")]
    df_spread<-spread(df_agg_to_spread,"dayhour","value")
  } else {
    df_agg_to_spread <- df_agg[,c("day","value","daypart")]
    df_agg_to_spread <- aggregate(data.frame("value"=df_agg_to_spread$value),by=list("day"=df_agg_to_spread$day,"daypart"=df_agg_to_spread$daypart),FUN=mean)
    df_spread<-spread(df_agg_to_spread,"daypart","value")
  }
  
  df_spread <- merge(df_spread,df_agg_d,by="day")
  days <- df_spread[,"day"]
  daily_cons <- df_spread[,"value"]
  daily_temp <- df_spread[,"temperature"]
  days_of_the_week <- as.numeric(strftime(days,"%u"))
  days_weekend <- ifelse(days_of_the_week %in% c(6,7),1,0)
  df_spread<-df_spread[,-which(colnames(df_spread) %in% c("value","day","temperature"))]
  
  if(perc_cons==T){load_curves <- normalize_perc_cons_day(df_spread)} else {load_curves <- df_spread}
  df_spread<- do.call(cbind,lapply(input_vars,function(x){eval(parse(text = x))}))
  normalization <- normalize_znorm(df_spread, norm_specs)#normalize_range_int(df_spread, 0, 1, norm_specs)
  df_spread <- normalization$norm
  
  # Generate the final spreated dataframe normalized
  complete_cases <- complete.cases(df_spread)
  if(filter_na==T){
    days <- days[complete_cases]
    df_spread <- df_spread[complete_cases,]
  }
  
  return(list("raw_df"= df_agg,"norm_df"=df_spread, "norm_specs"=normalization$specs, "perc_cons"=perc_cons, "n_dayparts"=n_dayparts,
              "input_vars"= input_vars, "days_complete"=days))
}

clustering_load_curves <- function(df, group, tz_local, time_column, value_column, temperature_column, perc_cons, n_dayparts, 
                                   norm_specs=NULL, input_vars, k=NULL, title="", output_plots=F, centroids_plot_file=NULL, bic_plot_file=NULL, folder_plots="",
                                   latex_font=F, plot_n_centroids_per_row=9, minimum_days_for_a_cluster=10, filename_prefix=NULL, force_plain_cluster=F){
  
  # df = df_
  # group = "tariff"
  # tz_local = "Europe/Madrid"
  # time_column = "time"
  # value_column = "powerContractM2"
  # temperature_column = "temperature"
  # k=2:6
  # perc_cons = T
  # n_dayparts = 6
  # norm_specs = NULL
  # input_vars = c("load_curves") # POSSIBLE INPUTS: c("load_curves", "days_weekend", "days_of_the_week", "daily_cons", "daily_temp"),
  # centroids_plot_file = "clustering.pdf"
  # bic_plot_file = "bic.pdf"
  # # centroids_plot_file = NULL
  # # bic_plot_file = NULL
  # latex_font = F
  # plot_n_centroids_per_row=2
  # minimum_days_for_a_cluster = 0
  # force_plain_cluster = F
  # filename_prefix=paste(postal_code,economic_sector,sep="~")
  
  # Only cluster days in periods without large weather dependence, without holidays and not during the covid lockdown
  if(sum(month(df[,time_column]) %in% 1:12)>10){
    df <- df[month(df[,time_column]) %in% c(4,5,9,10),]
  }
  # df <- df[!(as.Date(df[,time_column]) %in% 
  #              list_of_holidays(national_days = national_days, days_list = df$day)),]
  df <- df[as.Date(df[,time_column]) < as.Date("2020-03-15") |
             as.Date(df[,time_column]) > as.Date("2020-06-21"),]
  
  input_clust <- norm_load_curves(df, tz_local, time_column, value_column, temperature_column, perc_cons, n_dayparts, norm_specs, input_vars)
  
  # Initialize the objects
  # Test a clustering from 2 to 10 groups, if k is NULL (default).
  if(is.null(k)) k=seq(2,12)
  if((nrow(input_clust$norm_df)/4)<max(k)) k<-2:(nrow(input_clust$norm_df)/4)
  # Clustering model
  #mclustICL(apply(df_spread_norm,1:2,as.numeric),G = k, modelNames = c("EVI","VEI","EEI","VVI"))
  mclust_results <- Mclust(apply(input_clust$norm_df,1:2,as.numeric),G = k, modelNames=c("VEI","VVI","VII","EII"))
  mclust_results[['training_init']] = input_clust$norm_df
  mclust_results[['value_column']] = value_column
  clustering <- list("cluster"=predict(mclust_results)$classification,"centers"=t(mclust_results$parameters$mean),"k"=mclust_results$G)
  
  # Delete those individuals far from its centroid.
  distance <-distmat(apply(input_clust$norm_df,1:2,as.numeric),clustering$centers)
  
  df_distances <- data.frame(
    "distance"=mapply(function(r){distance[r,clustering$cluster[r]]},1:nrow(input_clust$norm_df)),
    "k"=clustering$cluster
  )
  
  for(j in unique(clustering$cluster)){
    if(max(df_distances[df_distances$k==j,"distance"])>2*quantile(df_distances[df_distances$k==j,"distance"],0)){
      print('yes')
      df_distances[df_distances$k==j,"k_new"] <- ifelse(
        (df_distances[df_distances$k==j,"distance"]>quantile(df_distances[df_distances$k==j,"distance"],1) |
           df_distances[df_distances$k==j,"distance"]<quantile(df_distances[df_distances$k==j,"distance"],0)),
        NA,
        j)
    } else {
      df_distances[df_distances$k==j,"k_new"] <- j
    }
  }
  
  # Generate the df_centroids dataset
  clustering_results <- data.frame(
    "day"=input_clust$days_complete,
    "s"=as.character(df_distances$k_new)
  )
  df_structural = merge(input_clust$raw_df, clustering_results, by="day")
  
  # # Deprecate those clusters with less than X days (X = minimum_days_for_a_cluster)
  # important_clusters <- names(table(df_structural$s)[table(df_structural$s) >= 24*minimum_days_for_a_cluster])
  # df_structural <- df_structural[df_structural$s %in% important_clusters,]
  # df_structural$s <- as.character(factor(df_structural$s, levels=unique(df_structural$s), labels=as.character(1:length(unique(df_structural$s)))))
  
  # Reformat the cluster integer numbers to factors of 2 digits
  df_structural$s <- sprintf("%02i",as.integer(as.character(df_structural$s)))
  
  # Compute the centroids
  df_centroids<-aggregate(df_structural$value,
                          by=list(df_structural$dayhour,df_structural$s),FUN=function(x){mean(x,na.rm=T)})#quantile(x,0.60,na.rm=T)
  colnames(df_centroids)<-c("dayhour","s","value")
  #df_centroids$s <- sprintf("%02i",as.integer(as.character(df_centroids$s)))
  df_centroids <- df_centroids[order(df_centroids$s),]
  df_centroids_spread <- spread(df_centroids,key = "dayhour",value = "value")
  
  # Compute the centroids
  df_centroids_avg<-data.frame(
    aggregate(df_structural$value, by=list(df_structural$dayhour), FUN=function(x){mean(x,na.rm=T)}),
    aggregate(df_structural$value, by=list(df_structural$dayhour), FUN=function(x){quantile(x,0.05,na.rm=T)})$x,
    aggregate(df_structural$value, by=list(df_structural$dayhour), FUN=function(x){quantile(x,0.95,na.rm=T)})$x
  )
  colnames(df_centroids_avg)<-c("dayhour","avg","lower","upper")
  
  # Add the flat cluster
  if(force_plain_cluster){
    flat_cluster <-c(sprintf("%02i",as.numeric(max(df_centroids$s))+1),rep(2,24))
    df_centroids_spread <- rbind(df_centroids_spread, flat_cluster)
  }
  
  # Plot
  if (!is.null(centroids_plot_file)){
    centroids_plot_file <- paste0(folder_plots,paste(filename_prefix, centroids_plot_file, sep="~"))
    print(paste0("Clustering results saved in: ",centroids_plot_file))
    if(latex_font==T){
      library(extrafont)
      #first download the fonts and unzip them  https://www.fontsquirrel.com/fonts/download/computer-modern
      # extrafont::font_import(pattern = "cmun*",paths = "/usr/share/fonts",recursive = T,prompt = F)
      loadfonts(quiet = T)
      pdf(centroids_plot_file,height = 8,width = 8)
      ggsave(centroids_plot_file, ggplot(df_structural) +
               geom_line(aes(x=as.numeric(substr(dayhour,1,2)),y=as.numeric(value),group=as.factor(day)),alpha= 0.2, col = 'black')+
               geom_line(data=df_centroids,aes(x=as.numeric(substr(dayhour,1,2)),y=as.numeric(value)),size=0.5,col='red')+
               facet_wrap(~as.numeric(s),nrow=ceiling(length(unique(df_centroids$s))/plot_n_centroids_per_row))+
               theme_bw()+theme(legend.position="none",axis.text=element_text(size=12),
                                axis.text.x = element_text(angle=90,vjust = 0.5, margin = margin(t=5, r=0, b = 10, l =0)),
                                axis.text.y = element_text(margin = margin(t=0, r= 10, b = 0, l = 10)),
                                axis.title=element_text(size=16))+labs(x="Hour of the day", y="W/m²") +
               theme(text= element_text(size=20, family="CM Roman"))+
               scale_x_continuous(
                 breaks = c(0, 12, 23),
                 label = c("00:00","12:00","23:00")
               ) + scale_y_continuous(#limits=c(min(df_centroids$value),quantile(df_centroids$value,1,na.rm = T)+0.1),
                 labels = if(all.equal(sum(rowSums(df_centroids_spread[,-1])),nrow(df_centroids_spread),tolerance=0.01)==T){scales::percent}else{scales::number})#scale_y_continuous(limits=c(min(df_centroids$value),max(df_centroids$value)+0.02),#quantile(df_centroids$value,0.99,na.rm = T)+0.1),
             #                   trans= "log2",
             #                   labels = scales::percent)
      )
      dev.off()
      # embed_fonts(file = centroids_plot_file, outfile=centroids_plot_file)
    } else {
      pdf(centroids_plot_file,height = 6.5,width = 6.5)
      print(ggplot(df_structural)+
              geom_line(aes(x=as.numeric(substr(dayhour,1,2)),y=as.numeric(value),group=as.factor(day)),alpha=0.2)+
              geom_line(data=df_centroids,aes(x=as.numeric(substr(dayhour,1,2)),y=as.numeric(value)),size=0.5,col='red')+
              facet_wrap(~s,nrow=ceiling(length(unique(df_centroids$s))/plot_n_centroids_per_row))+
              theme_bw()+theme(legend.position="none",axis.text=element_text(size=12),axis.text.x = element_text(angle=90),
                               axis.title=element_text(size=16))+labs(x="Hour of the day", y="W/m²") +
              theme(text= element_text(size=16))+
              scale_x_continuous(
                breaks = c(0, 12, 23),
                label = c("00:00","12:00","23:00")
              )  + scale_y_continuous(#limits=c(min(df_centroids$value),quantile(df_centroids$value,1,na.rm = T)+0.1),
                labels =if(all.equal(sum(rowSums(apply(df_centroids_spread[,-1],1:2,as.numeric))),
                                     nrow(df_centroids_spread),tolerance=0.01)==T){scales::percent}else{scales::number})#scale_y_log10(limits=c(min(df_centroids$value),max(df_centroids$value)+0.2),#c(0,quantile(df_centroids$value,0.99,na.rm = T)+0.1),
            #                    labels = scales::percent)
      )
      dev.off()
    }
  }
  
  bic_df <- data.frame(k,BIC=mclust_results$BIC[,])
  if (!is.null(bic_plot_file)){
    bic_plot_file <- paste0(folder_plots,paste(filename_prefix, bic_plot_file,sep="~"))
    print(paste0("BIC results saved in: ",bic_plot_file))
    if(latex_font==T){
      library(extrafont)
      #first download the fonts and unzip them  https://www.fontsquirrel.com/fonts/download/computer-modern
      # extrafont::font_import(pattern = "cmun*",paths = "/usr/share/fonts",recursive = T,prompt = F)
      loadfonts(quiet = T)
      pdf(bic_plot_file,height = 4,width = 6)
      if (!("BIC" %in% colnames(bic_df))){ bic_df$BIC <- bic_df[,paste0("BIC.",mclust_results$modelName)] }
      print(ggplot(bic_df)+
              geom_line(aes(k,BIC)) + geom_point(aes(k,BIC)) +
              geom_point(aes(k,BIC), col=2, cex=4, data=bic_df[which.max(bic_df$BIC),])+
              geom_text(aes(x = k,
                            y = BIC - (BIC-bic_df[which.min(bic_df$BIC),"BIC"])*0.15, 
                            label = paste("k =",k)),
                        size=(5/14)*25, family="CM Roman",
                        data=bic_df[which.max(bic_df$BIC),]) +
              theme_bw() + theme(text= element_text(size=30, family="CM Roman")))
      dev.off()
      # embed_fonts(file = bic_plot_file, outfile=bic_plot_file)
    } else {
      pdf(bic_plot_file,height = 4,width = 6)
      if (!("BIC" %in% colnames(bic_df))){ bic_df$BIC <- bic_df[,paste0("BIC.",mclust_results$modelName)] }
      print(ggplot(bic_df)+
              geom_line(aes(k,BIC)) + geom_point(aes(k,BIC)) +
              geom_point(aes(k,BIC), col=2, cex=4, data=bic_df[which.max(bic_df$BIC),])+
              geom_text(aes(x = k,
                            y = BIC - (BIC-bic_df[which.min(bic_df$BIC),"BIC"])*0.15, 
                            label = paste("k =",k)),
                        size=(5/14)*25,
                        data=bic_df[which.max(bic_df$BIC),]) +
              theme_bw() + theme(text= element_text(size=20)))
      dev.off()
    }
  }
  
  df_structural <- df_structural[order(df_structural[,"time"]),]
  
  # Simple calendar classification model by day week
  df_structural[,"dayweek"] <- strftime(df_structural[,"time"],"%u")
  df_structural[,"weekend"] <- ifelse(strftime(df_structural[,"time"],"%u") %in% c("6","7"),1,0)
  
  if(sum(df_structural$day >= as.Date("2020-06-21")-years(1))>0){
    if (length(unique(df_structural$s[df_structural$day >= as.Date("2020-06-21")-years(1)]))){
      mod_calendar <- multinom(formula = as.factor(s) ~ 0 + as.factor(dayweek), data = df_structural[df_structural$day >= as.Date("2020-06-21")-years(1),])
    } else {
      mod_calendar <- unique(df_structural$s[df_structural$day >= as.Date("2020-06-21")-years(1)])
    }
  } else {
    mod_calendar <- NULL
  }
  
  # df_centroids_spread[,group] <- gr
  # df_centroids_avg[,group] <- gr
  
  results_all <- list("df"=df_structural, "classified"=clustering_results, "centroids"=df_centroids_spread, "centroids_avg"=df_centroids_avg, "mod"=mclust_results, 
                      "norm_specs"=input_clust$norm_specs, "perc_cons"=input_clust$perc_cons, "n_dayparts"=input_clust$n_dayparts, "mod_calendar"=mod_calendar,
                      "input_vars"= input_clust$input_vars)

  # names(results_all) <- names(df_s)
  
  return(results_all)
}

classifier_load_curves <- function(df, df_centroids, clustering_mod, clustering_mod_calendar, tz_local, time_column, value_column, temperature_column, 
                                   perc_cons, n_dayparts, filename_prefix, norm_specs=NULL, input_vars, plot_n_centroids_per_row=2, plot_file=NULL, folder_plots=""){
  
  # df_centroids <- do.call(rbind, lapply(names(clustering),function(i){clustering[[i]]$centroids}))
  # df = df_
  # group = "tariff"
  # clustering_mod = setNames(lapply(names(clustering),function(i)clustering[[i]][["mod"]]),names(clustering))
  # clustering_mod_calendar = setNames(lapply(names(clustering),function(i)clustering[[i]][["mod_calendar"]]),names(clustering))
  # df_centroids = df_centroids[,!(colnames(df_centroids) %in% c("s"))]
  # tz_local = "Europe/Madrid"
  # time_column = "time"
  # value_column = "powerContractM2"
  # temperature_column = "temperature"
  # perc_cons = setNames(lapply(names(clustering),function(i)clustering[[i]][["perc_cons"]]),names(clustering))
  # n_dayparts = setNames(lapply(names(clustering),function(i)clustering[[i]][["n_dayparts"]]),names(clustering))
  # norm_specs = setNames(lapply(names(clustering),function(i)clustering[[i]][["norm_specs"]]),names(clustering))
  # input_vars = setNames(lapply(names(clustering),function(i)clustering[[i]][["input_vars"]]),names(clustering))
  # plot_n_centroids_per_row = 2
  # plot_file = "classification.pdf"
  # filename_prefix=paste(postal_code,economic_sector,sep="~")
  
  df_centroids_ini <- df_centroids
  
  input_class <- norm_load_curves(df, tz_local, time_column, value_column, temperature_column, perc_cons, 
                                  n_dayparts, norm_specs, input_vars,filter_na = F)
  
  # Wide format of the consumption dataframe
  dt <- with_tz(df[,time_column],tz=tz_local)
  df$hour <- hour(dt)
  df$date <- as.Date(dt)
  df <- df[!duplicated(df[,c("hour","date")]),]
  df_consumption_wp <- spread(as.data.frame(df[c("date","hour",value_column)]),"hour",value_column)
  
  # Contracts
  days <- df_consumption_wp[,"date"]
  df_consumption_wp <- df_consumption_wp[,-1]
  for(dayh in as.character(0:23)[!(as.character(0:23) %in% colnames(df_consumption_wp))]){
    df_consumption_wp[,dayh] <- 0
  }
  
  df_consumption_wp <- df_consumption_wp[,order(as.numeric(colnames(df_consumption_wp)))]
  
  # Classify across all centroids (the classification is done by calculating the cross distance matrix between the centroids data frame and the consumption data frame)
  dist_shapes <- t(proxy::dist(apply(as.matrix(df_centroids),1:2,as.numeric),
                               as.matrix(df_consumption_wp)))
  
  # shapes_1 contains the number of the centroid which is nearer to each daily consumption
  # shapes_2 contains the number of the centroid which folllows in distance...
  input_class$norm_df[is.na(input_class$norm_df)] <- 0
  
  shapes_1_classification <- data.frame(
    "days" = days,
    "classification" = dist_shapes %>% apply(1, function(x) {ifelse(sum(is.na(x))>1, NA, order(x)[1])})
  )
  predict_classif_load_curve <- predict(clustering_mod,input_class$norm_df)$classification
  if(!is.null(clustering_mod_calendar)){
    if(class(clustering_mod_calendar)=="character"){
      predict_classif_calendar <- clustering_mod_calendar
    } else {
      predict_classif_calendar <- predict(clustering_mod_calendar,
                                          data.frame(
                                            "dayweek"=strftime(input_class$days_complete,"%u")
                                          ))
    }
  } else {
    predict_classif_calendar <- predict_classif_load_curve
  }
  
  shapes_1_prediction <- data.frame(
    "days" = input_class$days_complete,
    "prediction" = 
      ifelse(
        input_class$days_complete >= as.Date("2020-03-15") &
          input_class$days_complete <= as.Date("2020-06-21"),
        as.character(as.numeric(as.character(predict_classif_calendar))),
        as.character(predict_classif_load_curve)
      )
  )
  
  shapes_1 <- merge(shapes_1_classification, shapes_1_prediction, by="days", all.x=T)
  shapes_1$final <- ifelse(is.na(shapes_1$prediction), shapes_1$classification, shapes_1$prediction)
  shapes_1 <- shapes_1$final
  shapes_2 <- dist_shapes %>% apply(1, function(x) {ifelse(sum(is.na(x))>1, NA, order(x)[2])})
  shapes_3 <- dist_shapes %>% apply(1, function(x) {ifelse(sum(is.na(x))>1, NA, order(x)[3])})
  
  dist_shapes <- data.frame(matrix(dist_shapes,ncol=ncol(dist_shapes)))
  colnames(dist_shapes) <- paste0("dist",1:ncol(dist_shapes))
  df_shapes <- data.frame(
    "day" = as.Date(days,format="%Y-%m-%d",
                    tz=tz_local),
    "shape1" = as.factor(shapes_1), "shape2" = as.factor(shapes_2), "shape3" = as.factor(shapes_3),
    dist_shapes)
  df_consumption <- df
  
  df_consumption$s <- df_shapes$shape1[match(df_consumption$date,df_shapes$day)]
  df_consumption[,"value"] <- df_consumption[,value_column]
  df_centroids_m <- reshape2::melt(data.frame("s"=1:nrow(df_centroids),df_centroids),"s")
  df_centroids_m$variable <- sprintf("%02i:00",as.integer(substr(df_centroids_m$variable,2,3)))
  colnames(df_centroids_m) <- c("s","dayhour","value")
  df_consumption_test <- df_consumption[ !is.na(df_consumption$s) , ]
  df_consumption_test$s <- as.numeric(df_consumption_test$s)
  
  if(!is.null(plot_file)){
    plot_file <- paste0(folder_plots,paste(filename_prefix, "classification.pdf",sep="~"))
    df_consumption_for_plot <- data.frame(
      "hour"=df_consumption_test[,"hour"],
      "date"=df_consumption_test[,"date"],
      "value"=df_consumption_test[,value_column],
      "s"=df_consumption_test[,"s"]
    )
    p <- ggplot(df_consumption_for_plot)+
      geom_line(aes(x=as.numeric(substr(hour,1,2)),y=as.numeric(value),group=as.factor(date)),alpha=0.2, col = 'black') +
      geom_line(data=df_centroids_m,aes(x=as.numeric(substr(dayhour,1,2)),y=as.numeric(value)),size=0.5,col='red') +
      facet_wrap(~s,nrow=ceiling(length(unique(df_centroids_m$s))/plot_n_centroids_per_row)) +
      theme_bw() + theme(legend.position="none",axis.text=element_text(size=12),axis.text.x = element_text(angle=90),
                         axis.title=element_text(size=16)) +
      labs(x="Hour of the day", y="W/m²") +
      theme(text= element_text(size=16))+
      scale_x_continuous(
        breaks = c(0, 12, 23),
        label = c("00:00","12:00","23:00")
      )
    ggsave(plot_file,p,height = 6.5,width = 6.5)
  }
  results_all <- df_consumption_test
    
  return(results_all)
  
}
