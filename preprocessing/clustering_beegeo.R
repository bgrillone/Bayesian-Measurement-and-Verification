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



## CHARACTERIZER FUNCTIONS ----


gaMonitor2 <- function (object, digits = getOption("digits"), ...){
  fitness <- na.exclude(object@fitness)
  cat(paste("GA | Iter =", object@iter, " | Mean =", format(mean(fitness), digits = digits),
            " | Best =", format(max(fitness), digits = digits),"\n"))
  flush.console()
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


rolling_dataset_by_n_months <- function(df, date_column = "day", n = 12, step = 3){
  
  ts_min <- min(df[(month(df[,date_column]-days(1)) %% step)==1,date_column],na.rm=T)
  ts_min <- as.Date(paste0(strftime(ts_min,"%Y-%m"),"-01"))
  ts_end <- max(df[,date_column],na.rm=T)
  
  df_l <- list()
  i <- 0
  while( (ts_min + months(n) - days(1)) <= ts_end || i==0){
    df_l[[paste(strftime(ts_min,"%Y%m"), strftime(ts_min + months(n) - days(1),"%Y%m"), sep="_")]] <- 
      df[df[,date_column] >= ts_min & df[,date_column] < (ts_min+months(n)),]
    nrows_df_aux <- nrow(df_l[[paste(strftime(ts_min,"%Y%m"), strftime(ts_min + months(n) - days(1),"%Y%m"), sep="_")]])
    if(nrow(df_l[[paste(strftime(ts_min,"%Y%m"), strftime(ts_min + months(n) - days(1),"%Y%m"), sep="_")]])>0){
      df_l[[paste(strftime(ts_min,"%Y%m"), strftime(ts_min + months(n) - days(1),"%Y%m"), sep="_")]]$weights <-
        (1:nrows_df_aux)^2/nrows_df_aux^2
    }
    step_i <- step
    for(step_i in step:1){
      if((ts_min + months(n+step_i) - days(1)) <= ts_end){break}
    }
    ts_min <- ts_min + months(step_i)
    i <- 1
  }
  
  df_l <- df_l[mapply(function(x) nrow(x)>0,df_l)]
  
  if(length(df_l)>0){
    df_l[["last_period"]] <- df_l[[names(df_l)[length(df_l)]]]
  }
  
  return(df_l)
}


aggregator <- function(df_ini, value_column="value", temperature_column = "temperature", windSpeed_column = "windSpeed", 
                       GHI_column = "GHI", time_column = "time", intercept_column="weekday_", tz_local="Europe/Madrid", group_column="season", hours_of_each_daypart=8){
  
  df <- df_ini
  
  # Add calendar variables to dataframe
  if (intercept_column=="weekday"){
    df[,"weekday"] <- factor(strftime(with_tz(df[,time_column],tz=tz_local),"%u"),
                             as.character(1:7),c("Mon","Tue","Wed","Thu","Fri","Sat","Sun"))
  }
  
  # GroupBy aggregation definition
  if(group_column %in% colnames(df)){
    df_agg <- data.frame("day"=as.Date(df[,time_column],tz=tz_local),"daypart"=as.character(floor(hour(with_tz(df[,time_column],tz=tz_local))/hours_of_each_daypart)),
                         "group"=df[,group_column])
    df[,c("day","daypart","group")] <- df_agg
    df$agg <- paste(df_agg[,"day"], df_agg[,"daypart"], df_agg[,"group"], sep="~")
  } else {
    df_agg <- data.frame("day"=as.Date(df[,time_column],tz=tz_local),"daypart"=as.character(floor(hour(with_tz(df[,time_column],tz=tz_local))/hours_of_each_daypart)))
    df[,c("day","daypart")] <- df_agg
    df$agg <- paste(df_agg[,"day"], df_agg[,"daypart"], sep="~")
    df$group <- "all"
  }
  
  # Aggregation of the initial data
  df <- smartAgg(df,by = "agg", 
                 function(x){mean(x,na.rm=T)}, c(value_column,temperature_column,windSpeed_column,GHI_column), 
                 function(x){x[1]}, c("day","daypart","group","s",intercept_column),
                 function(x){min(x,na.rm=T)}, "time",
                 catN = F)
  
  # Split the results by the group_column
  df_s <- split(df, df[,"group"])
  grs <- names(df_s)
  
  # Clear the outliers
  df_s <- lapply(grs, function(gr){
    df_aux <- df_s[[gr]]
    df_aux <- suppressMessages(pad(df_aux, by="day"))
    lend <- as.integer((365/2)* 24/hours_of_each_daypart)
    if (value_column %in% colnames(df_aux)){
      df_aux$f_v_r <- rollapply(df_aux[,value_column], width=lend, align="right", partial=T, FUN=function(x){max(lend-length(x),sum(is.na(x),na.rm=T))})
      df_aux$f_v_l <- rollapply(df_aux[,value_column], width=lend, align="left", partial=T, FUN=function(x){max(lend-length(x),sum(is.na(x),na.rm=T))})
      df_aux$f_t_r <- rollapply(df_aux[,temperature_column], width=lend, align="right", partial=T, FUN=function(x){max(lend-length(x),sum(is.na(x),na.rm=T))})
      df_aux$f_t_l <- rollapply(df_aux[,temperature_column], width=lend, align="left", partial=T, FUN=function(x){max(lend-length(x),sum(is.na(x),na.rm=T))})
      df_aux$fault <- (df_aux$f_v_r>(30 * 24/hours_of_each_daypart) & df_aux$f_v_l>(30 * 24/hours_of_each_daypart)) | (df_aux$f_t_r>(30 * 24/hours_of_each_daypart) & df_aux$f_t_l>(30 * 24/hours_of_each_daypart))
      med <- rollapply(df_aux[,value_column], width=(30 * 24/hours_of_each_daypart), align="center", partial=T, FUN=mean, na.rm=T)
      stddev <- rollapply(df_aux[,value_column], width=(30 * 24/hours_of_each_daypart), align="center", partial=T, FUN=sd, na.rm=T)
      df_aux$outliers <- (df_aux[,value_column]-med)/stddev
      df_aux$weights <- 1#ifelse(df_aux$outliers<1,1,(1/df_aux$outliers))
      df_aux$outliers <- df_aux$outliers > 3 
      df_aux <- df_aux[ !df_aux$fault & is.finite(df_aux$day) & is.finite(df_aux[,temperature_column]) & is.finite(df_aux[,value_column]),
                        !(colnames(df_aux) %in% c("agg", "f_v_r","f_v_l","f_t_r","f_t_l","fault","outliers","weights"))]
    } else {
      df_aux$f_t_r <- rollapply(df_aux[,temperature_column], width=lend, align="right", partial=T, FUN=function(x){max(lend-length(x),sum(is.na(x),na.rm=T))})
      df_aux$f_t_l <- rollapply(df_aux[,temperature_column], width=lend, align="left", partial=T, FUN=function(x){max(lend-length(x),sum(is.na(x),na.rm=T))})
      df_aux$fault <- (df_aux$f_t_r>(30 * 24/hours_of_each_daypart) & df_aux$f_t_l>(30 * 24/hours_of_each_daypart))
      df_aux <- df_aux[ !df_aux$fault & is.finite(df_aux$day) & is.finite(df_aux[,temperature_column]),
                        !(colnames(df_aux) %in% c("agg", "f_v_r","f_v_l","f_t_r","f_t_l","fault","outliers","weights")) ]
    }
    df_aux$daypart <- as.factor(as.character(df_aux$daypart))
    df_aux
  })
  names(df_s) <- grs
  
  df_s <- df_s[mapply(function(x){nrow(x)},df_s)>0]
  
  if(identical(grs,"all")){
    return(df_s$all)
  } else {
    return(df_s)
  }
}


mongo_conn <- function(collection, config){
  
  if(!("user" %in% names(config$mongo)) || config$mongo$user==""){
    mongo(collection = collection,
          url = sprintf("mongodb://%s:27017",
                        config$mongo$host),
          db = "beegeo"
    )
  } else {
    mongo(collection = collection,
          url = sprintf("mongodb://%s:%s@%s:27017",
                        config$mongo$user,
                        config$mongo$password,
                        config$mongo$host),
          db = "beegeo"
    )
  }
  
}


characterizer <- function(df_ini, value_column="value", temperature_column = "temperature", windSpeed_column = "windSpeed", 
                          GHI_column = "GHI", time_column = "time", intercept_column="weekday", tz_local="Europe/Madrid", 
                          group_column="season", hours_of_each_daypart=8, centroids = df_centroids, centroids_summary = df_centroids_avg,
                          classification = classification[,c("date","s","tariff")]){
  
  # df_ini = df_
  # tz_local = "Europe/Madrid"
  # time_column = "time"
  # value_column = "powerContractM2"
  # temperature_column = "temperature"
  # GHI_column = "GHI"
  # intercept_column = "weekday"
  # windSpeed_column = "windSpeed"
  # group_column ="tariff"
  # hours_of_each_daypart = 4
  # centroids = df_centroids
  # centroids_summary = df_centroids_avg
  # classification = classification[,c("date","s","tariff")]
  
  df_s <- df_ini
  
  if(class(df_s)=="data.frame"){ df_s <- list("all"=df_s)}
  
  # Run the trainer of the model for a one-year-window rolling dataset by months
  #gr <- names(df_s)[1]
  daily_df <- df_s
  
  daily_df_roll_years <- rolling_dataset_by_n_months(df = daily_df, date_column = "local_date", n = 12)
  
  results_year <- list()
  
  features <- do.call(c,list(
    lapply(1:(24/hours_of_each_daypart),FUN=function(x){list(min=12,max=25,n=15,class="float")}),
    lapply(1:(24/hours_of_each_daypart),FUN=function(x){list(min=0,max=7,n=15,class="float")}),
    lapply(1:length(levels(daily_df$s)),FUN=function(x){list(min=0,max=1,n=1,class="float")}),
    list(list(min=1,max=100,n=31,class="float")),
    list(list(min=0,max=1,n=1,class="int"))
  ))
  names(features) <- c(
    mapply(1:(24/hours_of_each_daypart),FUN=function(x){paste0("tbal_",x)}),
    mapply(1:(24/hours_of_each_daypart),FUN=function(x){paste0("hysteresis_",x)}),
    mapply(1:length(levels(daily_df$s)),FUN=function(x){paste0("seasonalities_wdep_",x)}),
    "thermal_time_constant",
    "training_without_holidays"
  )
  # tbal_min <- rep(10,(24/hours_of_each_daypart)) #*length(unique(daily_df$s))
  # tbal_max <- rep(30,(24/hours_of_each_daypart))
  # hysteresis_min <- rep(0,(24/hours_of_each_daypart))
  # hysteresis_max <- rep(7,(24/hours_of_each_daypart))
  # seasonalities_wdep_min <- rep(0,length(levels(daily_df$s)))
  # seasonalities_wdep_max <- rep(1,length(levels(daily_df$s)))
  
  y1 <- T
  
  if(identical(daily_df_roll_years,list())){
    return(NULL)
  }
  
  for (y in names(daily_df_roll_years)){
    #y <- names(daily_df_roll_years)[1]#14
    daily_df_year <- daily_df_roll_years[[y]]
    rows_to_train <- do.call(c,lapply(unique(daily_df_year$s),
                                      FUN= function(x){
                                        sample(as.numeric(rownames(daily_df_year[daily_df_year$s==x,])),
                                               nrow(daily_df_year[daily_df_year$s==x,])*1,replace = F)
                                      }))
    
    # Recursive Least Squares
    # GA <- ga(type = "real-valued",
    #          fitness = characterization_rls_model_trainer,
    #          lower = c(tbal_min,hysteresis_min), upper = c(tbal_max,hysteresis_max),
    #          df = daily_df_year,
    #          temperature_column = temperature_column,
    #          windSpeed_column = windSpeed_column,
    #          GHI_column = GHI_column,
    #          value_column = value_column,
    #          intercept_column = intercept_column,
    #          time_column = time_column,
    #          daypart_column = "daypart",
    #          for_optimize = T,
    #          hours_of_each_daypart = hours_of_each_daypart,
    #          rows_to_train = rows_to_train,
    #          monitor=gaMonitor2,
    #          maxiter=if(y1==T){100}else {10},popSize = 64,parallel= F,pmutation = 0.1)
    
    # Least squares
    # suggestions <- if(y1==T){
    #   t(data.frame(
    #     c(tbal_min,hysteresis_min,seasonalities_wdep_min),
    #     c(tbal_max,hysteresis_max, seasonalities_wdep_max)))
    # }else{
    #   t(data.frame(
    #     c(tbal,hysteresis,seasonalities_wdep),
    #       c(tbal_min,hysteresis_min,seasonalities_wdep_min),
    #       c(tbal_max,hysteresis_max,seasonalities_wdep_max))
    #     )
    # }
    
    GA <- ga(#type = "real-valued",
      type = "binary",
      fitness = characterization_model_trainer,
      # lower = c(tbal_min,hysteresis_min,seasonalities_wdep_min), 
      # upper = c(tbal_max,hysteresis_max,seasonalities_wdep_max),
      nBits = sum(mapply(function(x) { nchar(toBin(x)) }, mapply(function(i){i[['n']]},features))),##MODEL
      min_per_feature = mapply(function(i){i[['min']]},features),##DATA TO RUN
      max_per_feature = mapply(function(i){i[['max']]},features),##DATA TO RUN
      nclasses_per_feature = mapply(function(i){i[['n']]},features),##DATA TO RUN
      class_per_feature = mapply(function(i){i[['class']]},features),##DATA TO RUN
      names_per_feature = names(features),##DATA TO RUN
      df = daily_df_year,
      temperature_column = temperature_column,
      windSpeed_column = windSpeed_column,
      GHI_column = GHI_column,
      value_column = value_column,
      intercept_column = intercept_column,
      time_column = time_column,
      daypart_column = "daypart",
      for_optimize = T,
      hours_of_each_daypart = hours_of_each_daypart,
      rows_to_train = NULL,#rows_to_train,
      monitor = gaMonitor2,
      suggestions = NULL,
      selection = gabin_tourSelection,##MODEL
      mutation = gabin_raMutation,
      crossover = partial(bee_uCrossover,nclasses_per_feature = mapply(function(i){i[['n']]},features)),
      optim = F,
      maxiter=if(y1==T){30}else{12},popSize = 144,parallel= 24,pmutation = 0.05) #monitor = monitor)
    params <- decodeValueFromBin(GA@solution[1,], 
                                 min_per_feature = mapply(function(i){i[['min']]},features),##DATA TO RUN
                                 max_per_feature = mapply(function(i){i[['max']]},features),##DATA TO RUN
                                 nclasses_per_feature = mapply(function(i){i[['n']]},features),##DATA TO RUN
                                 class_per_feature = mapply(function(i){i[['class']]},features))##DATA TO RUN)
    names(params) <- names(features)
    tbal <- params[grepl("^tbal",names(params))]
    hysteresis <- params[grepl("^hysteresis",names(params))]
    seasonalities_wdep <- params[grepl("^seasonalities_wdep",names(params))]
    thermal_time_constant <- params["thermal_time_constant"]
    training_without_holidays <- params["training_without_holidays"]
    
    for(tbal_item in names(features)[grepl("^tbal",names(features))]){
      features[[tbal_item]]$min <- tbal[as.numeric(gsub("tbal_","",tbal_item))] - 2
      features[[tbal_item]]$max <- tbal[as.numeric(gsub("tbal_","",tbal_item))] + 2
      features[[tbal_item]]$n <- 3
    }
    for(hysteresis_item in names(features)[grepl("^hysteresis",names(features))]){
      features[[hysteresis_item]]$min <- 
        if((hysteresis[as.numeric(gsub("hysteresis_","",hysteresis_item))]-0.5)>0){
          hysteresis[as.numeric(gsub("hysteresis_","",hysteresis_item))] - 0.5
        } else {0}
      features[[hysteresis_item]]$max <- 
        if((hysteresis[as.numeric(gsub("hysteresis_","",hysteresis_item))]+0.5)<7){
          hysteresis[as.numeric(gsub("hysteresis_","",hysteresis_item))] +0.5
        } else {7}
      features[[hysteresis_item]]$n <- 3
    }
    for(seasonalities_wdep_item in names(features)[grepl("^seasonalities_wdep",names(features))]){
      features[[seasonalities_wdep_item]]$min <- seasonalities_wdep[as.numeric(gsub("seasonalities_wdep_","",seasonalities_wdep_item))]
      if (seasonalities_wdep[as.numeric(gsub("seasonalities_wdep_","",seasonalities_wdep_item))] == 1){
        features[[seasonalities_wdep_item]]$n <- 0
      }
    }
    features[["thermal_time_constant"]]$min <- thermal_time_constant - 6 
    features[["thermal_time_constant"]]$max <- thermal_time_constant + 6
    if(features[["thermal_time_constant"]]$min<1){
      features[["thermal_time_constant"]]$min <- 1
    }
    if(features[["thermal_time_constant"]]$max>100){
      features[["thermal_time_constant"]]$max <- 100 
    }
    features[["thermal_time_constant"]]$n <- 7
    
    daily_model <- characterization_model_trainer(
      params = c(tbal,hysteresis,seasonalities_wdep,thermal_time_constant,training_without_holidays),
      temperature_column = temperature_column,
      windSpeed_column = windSpeed_column,
      GHI_column = GHI_column,
      value_column = value_column,
      intercept_column = intercept_column,
      time_column = time_column,
      for_optimize = F,
      daypart_column = "daypart",
      df = daily_df_year,
      hours_of_each_daypart = hours_of_each_daypart,
      rows_to_train = NULL)
    
    # daily_model <- characterization_rls_model_trainer(
    #   params = c(tbal,hysteresis), 
    #   temperature_column = temperature_column,
    #   windSpeed_column = windSpeed_column,
    #   GHI_column = GHI_column,
    #   value_column = value_column,
    #   intercept_column = intercept_column,
    #   time_column = time_column,
    #   for_optimize = F,
    #   daypart_column = "daypart",
    #   df = daily_df_year,
    #   hours_of_each_daypart = hours_of_each_daypart)
    
    # coef(daily_model$mod)
    # plotly::ggplotly(ggplot(daily_model$df)+geom_line(aes(time,powerContractM2))+geom_line(aes(time,pred),col="red",alpha=0.5))
    # plotly::ggplotly(ggplot(daily_model$df)+geom_line(aes(time,powerContractM2))+geom_line(aes(time,pred_ini),col="red",alpha=0.5)+geom_line(aes(time,holidays_component),col="blue",alpha=0.5)+
    #                    geom_point(aes(time,holidays_component,col=s),alpha=0.5))
    # ggplot(daily_model$df) + geom_point(aes(time,powerContractM2,col=holidays))
    
    daily_model$df[,group_column] <- daily_model$df$group
    daily_model$df$rollYear <- y
    #daily_model$summary[,group_column] <- gr
    #daily_model$summary$rollYear <- y
    
    results_year[[y]] <- list("df"=daily_model$df,
                              #"summary"=daily_model$summary,
                              "mod"=daily_model$mod)
    y1 <- F
    
  }
  
  names(results_year) <- names(daily_df_roll_years)
  
  gc(reset = T,full = T,verbose = F)
  
  list(
    "df"=do.call(rbind,lapply(results_year,function(i){i$df})),
    #"summary"=do.call(rbind,lapply(results_year,function(i){i$summary})),
    "mod"=lapply(results_year,function(i){i$mod})
  )
    
  gc(reset = T,full = T,verbose = F)
  
  # Filter all tariffs which have NULL results
  filter_results <- mapply(function(i)is.null(i),results_all)
  results_all <- results_all[!filter_results]
  if(identical(list(),results_all)){
    return(NULL)
  }
  
  names(results_all) <- names(df_s)
  
  # Prediction dataframe output
  results_all <- list(
    "df" = do.call(rbind, lapply(FUN=function(i){i$df},results_all)),
    #"summary" = do.call(rbind, lapply(FUN=function(i){i$summary},results_all)),
    "mod" = lapply(FUN=function(i){i$mod},results_all)
  )
  
  results_all$df[,c("iniDate","endDate")] <- do.call(rbind,strsplit(results_all$df$rollYear,split = "_"))
  results_all$df$iniDate <- ifelse(results_all$df$rollYear=="last_period",NA,as.Date(paste0(results_all$df$iniDate,"01"),"%Y%m%d"))
  results_all$df$endDate <- ifelse(results_all$df$rollYear=="last_period",NA,as.Date(paste0(results_all$df$endDate,"01"),"%Y%m%d") + months(1) - days(1))
  
  # Config aggregator output
  results_all$config_aggregator <- list(
    tz_local = tz_local,
    time_column = time_column,
    value_column = value_column,
    temperature_column = temperature_column,
    GHI_column = GHI_column,
    intercept_column = intercept_column,
    windSpeed_column = windSpeed_column,
    group_column =group_column,
    hours_of_each_daypart = hours_of_each_daypart
  )
  
  # Daypart values output
  daypart_values <- data.frame(
    "hour"=0:23,
    "dayPart"=rep(0:((24/hours_of_each_daypart)-1),each=hours_of_each_daypart)
  )
  daypart_values <- smartAgg(daypart_values,by = "dayPart",function(i)min(i),"hour",function(i)max(i),"hour",catN = F)
  colnames(daypart_values) <- c("dayPart","minH","maxH")
  daypart_values$hourLabel <- paste0(daypart_values$minH,"-",daypart_values$maxH,"h")
  results_all$daypart_values <- daypart_values
  
  # Clustering and classification output
  results_all$centroids <- centroids
  results_all$centroids_summary <- centroids_summary
  results_all$classification <- classification
  
  return(results_all)
}

bee_uCrossover <- function(object, parents, nclasses_per_feature){
  parents <- object@population[parents, , drop = FALSE]
  u <- unlist(lapply(nclasses_per_feature, function(i){rep(runif(1),nchar(toBin(i)))}))
  children <- parents
  children[1:2, u > 0.5] <- children[2:1, u > 0.5]
  out <- list(children = children, fitness = rep(NA, 2))
  return(out)
}

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

tune_with_params <- function(df, params, temperature_column, GHI_column, windSpeed_column, daypart_column, weekday_column, time_column, 
                             seasonality_column = "s", force_weather_heating_vars_to_0 = F, force_weather_cooling_vars_to_0 = F, 
                             hours_of_each_daypart=8, training_seasonality_levels=NULL, force_covid_lockdown=NULL){
  
  # Assign holidays
  holidays <- list_of_holidays(national_days = national_days, days_list = df$day)
  df$holidays <- ifelse(df$day %in% holidays,1,0)
  # The holidays consumption affectation starts aprox 12 hours before
  df$holidays <- pmax(
    lead(df$holidays,ceiling(8/hours_of_each_daypart),df$holidays[nrow(df)]),
    df$holidays)
  
  # Force seasonality factors
  if(!is.null(training_seasonality_levels)){
    df[,seasonality_column] <- factor(df[,seasonality_column], levels=training_seasonality_levels)
  }
  
  # Weather transformations
  tbal <- params[grepl("^tbal",names(params))]
  hysteresis <- params[grepl("^hysteresis",names(params))]
  seasonalities_wdep <- round(params[grepl("^seasonalities_wdep",names(params))],0)
  thermal_time_constant <- params["thermal_time_constant"]
  
  df <- add_cooling_heating_daily_deltaT_features(
    df = df,temperature_column = temperature_column, tbal = tbal, hysteresis = hysteresis,
    seasonalities_wdep = seasonalities_wdep, daypart_column = daypart_column, thermal_time_constant = thermal_time_constant,
    hours_of_each_daypart = hours_of_each_daypart
  )
  df <- add_cooling_heating_daily_GHI_features(
    df = df,temperature_column = temperature_column, GHI_column = GHI_column
  )
  df <- add_cooling_heating_daily_windSpeed_features(
    df = df,temperature_column = temperature_column, windSpeed_column = windSpeed_column
  )
  df <- add_fs_daypart(
    df = df, daypart_column = daypart_column,hours_of_each_daypart = hours_of_each_daypart
  )
  df <- add_fs_weekpart(
    df = df, daypart_column = daypart_column, weekday_column= weekday_column, hours_of_each_daypart = hours_of_each_daypart
  )
  df <- add_fs_daypart_holidays(
    df = df, daypart_column = daypart_column,hours_of_each_daypart = hours_of_each_daypart
  )
  df <- add_fs_yearpart_holidays(
    df = df, time_column = time_column
  )
  if (force_weather_heating_vars_to_0==T) {
    df[,grepl(paste(paste0(temperature_column,'_h'), paste0(temperature_column,'_status_h'), 
                    paste0(temperature_column,'_lp_h'), paste0(windSpeed_column,'_h'), 
                    paste0(GHI_column,'_h'), sep="|"), 
              colnames(df))] <- 0
  }
  if (force_weather_cooling_vars_to_0==T) {
    df[,grepl(paste(paste0(temperature_column,'_c'), paste0(temperature_column,'_status_c'), 
                    paste0(temperature_column,'_lp_c'), paste0(windSpeed_column,'_c'), 
                    paste0(GHI_column,'_c'), sep="|"),
              colnames(df))] <- 0
  }
  
  # Generate all the seasonal dummies
  df <- cbind(df,as.data.frame(model.matrix(~ 0 + s, df)))
  
  # Covid lockdown period
  if(!is.null(force_covid_lockdown)){
    df$covid_lockdown <- factor(rep(force_covid_lockdown,nrow(df)), levels=c(T,F))
  } else {
    df$covid_lockdown <- as.factor(df$time >= as.POSIXct("2020-03-15 00:00:00", tz="Europe/Madrid") &
                                     df$time <= as.POSIXct("2020-06-21 00:00:00", tz="Europe/Madrid"))
  }
  
  # Tune covid seasonal dummies
  seasonalities_covid <- df[colnames(df)[grepl("^s[0-9]{2}$",colnames(df))]] * ifelse(as.logical(as.character(df$covid_lockdown)),1,0)
  colnames(seasonalities_covid) <- paste0(colnames(seasonalities_covid), "_covid_lockdown")
  df <- cbind(
    df,
    seasonalities_covid)  
  
  # Tune covid daypart fourier series 
  dp_covid <- df[colnames(df)[grepl("^daypart_fs",colnames(df))]] * ifelse(as.logical(as.character(df$covid_lockdown)),1,0)
  colnames(dp_covid) <- paste0("covid_lockdown_",colnames(dp_covid))
  df <- cbind(
    df,
    dp_covid)
  
  return(df)
}

characterization_rls_model_trainer <- function(params, df, temperature_column = "temperature", value_column = "value", windSpeed_column = "windSpeed", 
                                               daypart_column = "daypart", time_column= "time", GHI_column = "GHI", intercept_column = "weekday", for_optimize=F, 
                                               hours_of_each_daypart=8,rows_to_train=NULL){
  error <- F
  tryCatch({
    # df <- daily_df_year
    if(!is.null(rows_to_train)){
      df$val <- !(rownames(df) %in% as.character(rows_to_train))
    } else {
      df$val <- F
    }
    df <- tune_with_params(df = df, params = params, temperature_column = temperature_column, GHI_column = GHI_column, 
                           windSpeed_column = windSpeed_column, daypart_column = daypart_column, weekday_column = intercept_column, 
                           time_column = time_column, seasonality_column = "s", hours_of_each_daypart = hours_of_each_daypart,
                           training_seasonality_levels = NULL)
    # RLS version
    form <- as.formula(sprintf("%s ~ 1 + %s"
                               ,value_column
                               ,paste(
                                 paste0(colnames(df)[grepl("^daypart_fs_",colnames(df))],"",collapse=" + ")
                                 , "temperature_c"
                                 , paste0(mapply(function(x)paste0("temperature_c:",x,""),colnames(df)[grepl("^daypart_fs_",colnames(df))]),collapse=" + ") #colnames(df)[grepl("^daypart_fs_",colnames(df))] #"as.factor(daypart)"
                                 , "temperature_h"
                                 , paste0(mapply(function(x)paste0("temperature_h:",x,""),colnames(df)[grepl("^daypart_fs_",colnames(df))]),collapse=" + ")#,temperature_column,daypart_column #colnames(df)[grepl("^daypart_fs_",colnames(df))] #"as.factor(daypart)"
                                 ,sep=" + ")
    ))
    y <- model.frame(form,df)[,1]
    x <- as.matrix(model.matrix(form,df))
    ist <-  24/hours_of_each_daypart*7
    mod <- RLS(y,x,ist = ist)
    ggplot(reshape2::melt(mod$beta)) + geom_line(aes(Var1,value)) + facet_wrap(~Var2,scales='free_y')
    x_real <- as.matrix(x)[(ist+1):nrow(x),]
    y_real <- y[(ist+1):nrow(x)]
    # plot(rowSums(as.matrix(mod$beta) * x_real),type="l")
    # lines(y_real,col="red")
  }, error=function(e) error<-T)
  
  if(for_optimize==T){
    
    if(error == T){
      return(-10000)
    } else {
      return(
        -rmserr(rowSums(as.matrix(mod$beta) * x_real),y_real)$rmse
      )
    }
    
  } else {
    df <- df[(ist+1):nrow(x),]
    df$pred <- rowSums(as.matrix(mod$beta) * x_real)
    return(list("df"=df, "mod"=mod))
  }
}


characterization_model_trainer <- function(X=NULL, params = NULL, nBits,min_per_feature,max_per_feature,nclasses_per_feature,class_per_feature,names_per_feature, df, temperature_column = "temperature", value_column = "value", windSpeed_column = "windSpeed", 
                                           daypart_column = "daypart", time_column= "time", GHI_column = "GHI", intercept_column = "weekday", for_optimize=F, 
                                           hours_of_each_daypart=8,rows_to_train=NULL){
  
  # nBits = sum(mapply(function(x) { nchar(toBin(x)) }, mapply(function(i){i[['n']]},features)))
  # min_per_feature = mapply(function(i){i[['min']]},features)
  # max_per_feature = mapply(function(i){i[['max']]},features)
  # nclasses_per_feature = mapply(function(i){i[['n']]},features)
  # class_per_feature = mapply(function(i){i[['class']]},features)
  # names_per_feature = names(features)
  # X=sample(0:1, nBits,replace=T)
  
  if(is.null(params)){
    params <- decodeValueFromBin(X, class_per_feature, nclasses_per_feature, min_per_feature = min_per_feature, 
                                 max_per_feature = max_per_feature)
    names(params) <- names_per_feature
  }
  
  
  tryCatch({
    
    #df <- daily_df_year
    #params <- c(c(16,16,16,16,16,16), c(2,2,2,2,2,2), c(1,1,1,1,1,1))
    if(!is.null(rows_to_train)){
      df$val <- !(rownames(df) %in% as.character(rows_to_train))
    } else {
      df$val <- F
    }
    df <- tune_with_params(df = df, params = params, temperature_column = temperature_column, GHI_column = GHI_column, 
                           windSpeed_column = windSpeed_column, daypart_column = daypart_column, weekday_column = intercept_column, 
                           time_column = time_column, seasonality_column = "s", hours_of_each_daypart = hours_of_each_daypart,
                           training_seasonality_levels = NULL)
    if(length(unique(df$covid_lockdown))==1){
      form <- as.formula(sprintf("%s ~ 0 + %s"
                                 ,value_column
                                 ,paste(
                                   "s"
                                   #"daypart:s"
                                   , paste0(colnames(df)[grepl("^daypart_fs_",colnames(df))],":s",collapse=" + ")
                                   # , paste0(colnames(df)[grepl("^weekpart_fs_",colnames(df))],":s",collapse=" + ")
                                   , "temperature_lp_c:daypart"
                                   , "temperature_c:daypart"
                                   , "temperature_lp_h:daypart"
                                   , "temperature_h:daypart"
                                   , "windSpeed_c:daypart"
                                   , "windSpeed_h:daypart"
                                   # , "as.factor(holidays)"
                                   # , paste0(
                                   #   apply(expand.grid(colnames(df)[grepl("^daypart_holidays_fs_",colnames(df))], colnames(df)[grepl("^yearpart_holidays_fs_",colnames(df))],stringsAsFactors = F),1,paste,collapse=":")
                                   #   ,collapse=" + ")
                                   #, paste0(colnames(df)[grepl("^daypart_holidays_fs_",colnames(df))],"",collapse=" + ")
                                   #, paste0(colnames(df)[grepl("^yearpart_holidays_fs_",colnames(df))],"",collapse=" + ")
                                   ,sep=" + ")
      ))
      if(params["training_without_holidays"]==0){
        form <- update.formula(form,
                               as.formula(
                                 sprintf(".~. + %s",
                                         paste0(
                                           paste0(colnames(df)[grepl("^weekpart_fs_",colnames(df))],":s",collapse=" + ")
                                           ,collapse=" + ")
                                 )
                               )
        )
      }
    } else {
      form <- as.formula(sprintf("%s ~ 0 + %s"
                                 ,value_column
                                 ,paste(
                                   #"daypart:s:covid_lockdown"
                                   paste0(colnames(df)[grepl("^s[0-9]{2}$",colnames(df))],collapse=" + ")
                                   ,paste0(colnames(df)[grepl("^s[0-9]{2}_covid_lockdown$",colnames(df))],collapse=" + ")
                                   # ,"s"
                                   , paste0(colnames(df)[grepl("^covid_lockdown_daypart_fs_",colnames(df))],":s",collapse=" + ")
                                   , paste0(colnames(df)[grepl("^daypart_fs_",colnames(df))],":s",collapse=" + ")
                                   # , paste0(colnames(df)[grepl("^weekpart_fs_",colnames(df))],":s:covid_lockdown",collapse=" + ")
                                   , "temperature_lp_c:daypart:covid_lockdown"
                                   , "temperature_c:daypart:covid_lockdown"
                                   , "temperature_lp_h:daypart:covid_lockdown"
                                   , "temperature_h:daypart:covid_lockdown"
                                   , "windSpeed_c:daypart:covid_lockdown"
                                   , "windSpeed_h:daypart:covid_lockdown"
                                   # , "as.factor(holidays):covid_lockdown"
                                   # , paste0(
                                   #   apply(expand.grid(colnames(df)[grepl("^daypart_holidays_fs_",colnames(df))], colnames(df)[grepl("^yearpart_holidays_fs_",colnames(df))],stringsAsFactors = F),1,paste,collapse=":")
                                   #   ,":covid_lockdown",collapse=" + ")
                                   # , paste0(colnames(df)[grepl("^daypart_holidays_fs_",colnames(df))],":covid_lockdown",collapse=" + ")
                                   # , paste0(colnames(df)[grepl("^yearpart_fs_",colnames(df))],":covid_lockdown",collapse=" + ")
                                   
                                   ,sep=" + ")
      ))
      if(params["training_without_holidays"]==0){
        form <- update.formula(form,
                               as.formula(
                                 sprintf(".~. + %s",
                                         paste0(
                                           paste0(colnames(df)[grepl("^weekpart_fs_",colnames(df))],":s",collapse=" + ")
                                           ,collapse=" + ")
                                 )
                               )
        )
      }
    }
    
    # LS version
    # mod2 <- lm(formula = form, data = df)
    if(params["training_without_holidays"]==1){
      filter_df <- df$val==F & df$holidays==0    
    } else {
      filter_df <- df$val==F
    }
    
    y <- model.frame(form,df[filter_df,])[,1]
    x <- as.matrix(model.matrix(form,df[filter_df,]))
    
    mod_pen <- tryCatch({
      penalized(
        y,x,~0,positive = grepl("temperature|GHI|windSpeed",colnames(x)), #^s[0-9]{2}$
        lambda1 = 0,lambda2 = 0, maxiter = 25,
        #startbeta = ifelse(grepl("temperature|GHI|windSpeed",colnames(x)),0,1),
        trace=F
      )
    }, error=function(e){
      return(-10000)
      # penalized(
      #   y,x,~0,positive = grepl("temperature|GHI|windSpeed",colnames(x)),
      #   lambda1 = 0.5,lambda2 = 0,
      #   #startbeta = ifelse(grepl("temperature|GHI|windSpeed",colnames(x)),0,1),
      #   trace=F, model="linear"
      # )
    })
    
    tryCatch({length(penalized::coefficients(mod_pen))},error=function(e){return(-10000)})
    
    # return(length(penalized::coef(mod_pen)))
    
    df$pred <- c(as.matrix(model.matrix(form[-2],df))[,names(penalized:::coefficients(mod_pen))] %*% penalized:::coefficients(mod_pen))
    df$pred_ini <- df$pred
    
    # Quantify the holidays component
    rmse_error_no_holidays <- rmserr(df[df$holidays==0,value_column],df$pred[df$holidays==0])$rmse
    rmse_error_holidays <- rmserr(df[df$holidays==1,value_column],df$pred[df$holidays==1])$rmse
    if((rmse_error_holidays >= (rmse_error_no_holidays * 1.2)) && params["training_without_holidays"]==1){
      difference_during_holidays <- ifelse(df$holidays==1,df[,value_column] - df$pred,0)
      df$holidays_component <- ifelse(abs(difference_during_holidays)/df$pred >= 0.05,difference_during_holidays,0)
      df$holidays_component <- ifelse(
        as.Date(df[,time_column]) >= as.Date("2020-03-15") &  
          as.Date(df[,time_column]) <= as.Date("2020-06-21"),
        0,
        df$holidays_component)
      df$pred <- df$pred + df$holidays_component
    } else {
      df$holidays_component <- 0
    }
    
    #df$pred2 <- predict(mod2,df)
    #plotly::ggplotly(ggplot(df)+geom_line(aes(time,powerContractM2))+geom_line(aes(time,pred),col="red",alpha=0.5)
    #)#+geom_line(aes(time,pred2),col="green",alpha=0.5))
    
  }, error=function(e){return(-20000)})
  
  if(for_optimize==T){
    
    # Filter the days without weather dependency
    if(params["training_without_holidays"]==1){
      dfv <- df[
        is.finite(df$pred) & is.finite(df[,value_column]),# & df$val==T,
      ]
    } else {
      dfv <- df[
        is.finite(df$pred) & is.finite(df[,value_column]),# & df$val==T,
      ]
    }
    if(is.numeric(dfv$pred) & is.numeric(dfv[,value_column])){
      return(
        -rmserr(dfv$pred,dfv[,value_column])$rmse
      )
    } else {
      return(-10000)
    }
  } else {
    mod_pen@nuisance$tbal <- params[grepl("^tbal",names(params))]
    mod_pen@nuisance$hysteresis <- params[grepl("^hysteresis",names(params))]
    mod_pen@nuisance$seasonalities_wdep <- round(params[grepl("^seasonalities_wdep",names(params))],0)
    mod_pen@nuisance$thermal_time_constant <- params["thermal_time_constant"]
    mod_pen@nuisance$training_without_holidays <- params["training_without_holidays"]
    mod_pen@nuisance$formula <- form
    mod_pen@nuisance$seasonality_levels <- levels(df[filter_df,"s"])
    mod_pen@nuisance$training_filter_df <- filter_df
    mod_pen@nuisance$model <- df
    return(list("df"=df, "mod"=mod_pen))
  }
  
}

lp_vector<- function(x, a1) {
  ## Make a 1'st order low pass filter as (5.3) p.46 in the HAN report.
  y <- numeric(length(x))
  ## First value in x is the init value
  y[1] <- x[1]
  ## 
  for (i in 2:length(x)) {
    if (is.na(y[i - 1])) {
      y[i] <- x[i]
    } else {
      y[i] <- a1 * y[i - 1] + (1 - a1) * x[i]
    }
  }
  ## Return (afterwards the init value y[1], must be handled)
  return(y)
}

lp_vector_2<- function(x, thermal_time_constant_in_hours, timesteps_per_hour=1) {
  ## Make a 1'st order low pass filter as (5.3) p.46 in the HAN report.
  y <- numeric(length(x))
  ## Alpha definition
  alpha <- 1-(exp(1)^(-timesteps_per_hour/((2*pi)*thermal_time_constant_in_hours/24)))# timesteps_per_hour/(thermal_time_constant_in_hours*timesteps_per_hour+timesteps_per_hour) #
  
  for (i in 1:length(x)) {
    if (is.na(y[i - 1]) || i==1) {
      y[i] <- (alpha) * x[i]
    } else {
      y[i] <- (1-alpha) * y[i - 1] + (alpha) * x[i]
    }
  }
  ## Return (afterwards the init value y[1], must be handled)
  return(y)
  
  ## Graphical explanation of low pass filters for multiple examples of time constants
  # examplev <- c(rep(0,20),rep(1,96))
  # example <- data.frame("t"=1:length(examplev)-sum(examplev==0),"temp"=examplev)
  # example$lp3 <- lp_vector_2(example$temp,3)
  # example$lp6 <- lp_vector_2(example$temp,6)
  # example$lp12 <- lp_vector_2(example$temp,12)
  # example$lp24 <- lp_vector_2(example$temp,24)
  # example$lp48 <- lp_vector_2(example$temp,48)
  # example$lp72 <- lp_vector_2(example$temp,72)
  # example_melted <- reshape2::melt(example,c("t","temp"))
  # example_melted$hours <- sprintf("%02i",as.integer(gsub("lp","",as.character(example_melted$variable))))
  # example_melted <- example_melted %>% arrange(as.numeric(hours))
  # ggplot(example_melted) +
  #   geom_line(aes(t,value,col=hours)) +
  #   geom_line(aes(t,temp)) + theme_minimal() + ylab("Input (black) / Output") +
  #   scale_color_discrete(name="Thermal time\nconstant (h)") + 
  #   theme(text=element_text(size=14))
}

add_cooling_heating_daily_deltaT_features <- function(df, temperature_column="temperature", daypart_column="daypart",
                                                      tbal=19, hysteresis=1, seasonalities_wdep, thermal_time_constant=12, hours_of_each_daypart=8){
  
  tbal_vector <- tbal[as.integer(as.character(df[,daypart_column]))+1]
  #tbal[(as.integer(as.character(df[,daypart_column]))+1)+max(as.integer(as.character(df[,daypart_column]))+1)*(as.numeric(as.character(df$s))-1)]
  hysteresis_vector <- hysteresis[as.integer(as.character(df[,daypart_column]))+1]
  #hysteresis[(as.integer(as.character(df[,daypart_column]))+1)+max(as.integer(as.character(df[,daypart_column]))+1)*(as.numeric(as.character(df$s))-1)]
  seasonalities_wdep_vector <- seasonalities_wdep[as.numeric(as.character(df$s))]==1 
  #lp_temperature_h <- lp_vector(ifelse((tbal_vector-hysteresis_vector) >= df[,temperature_column], (tbal_vector-hysteresis_vector) - df[,temperature_column], 0), alpha_temperature)
  #lp_temperature_c <- lp_vector(ifelse((tbal_vector+hysteresis_vector) <= df[,temperature_column], df[,temperature_column] - (tbal_vector+hysteresis_vector), 0), alpha_temperature)
  lp_temperature <- lp_vector_2(df[,temperature_column],thermal_time_constant,1/hours_of_each_daypart)
  lp_temperature_h <- ifelse((tbal_vector-hysteresis_vector) >= lp_temperature, (tbal_vector-hysteresis_vector) - lp_temperature, 0)
  lp_temperature_c <- ifelse((tbal_vector+hysteresis_vector) <= lp_temperature, lp_temperature - (tbal_vector+hysteresis_vector), 0)
  
  df[,paste0(temperature_column,"_h")] <- ifelse(
    ((tbal_vector-hysteresis_vector) >= df[,temperature_column]) & (seasonalities_wdep_vector==T | df$holidays==T),
    (tbal_vector-hysteresis_vector) - df[,temperature_column], 0)
  df[,paste0(temperature_column,"_lp_h")] <- ifelse((tbal_vector-hysteresis_vector) >= df[,temperature_column] & (seasonalities_wdep_vector==T | df$holidays==T), 
                                                    lp_temperature_h, 0)
  df[,paste0(temperature_column,"_status_h")] <- ifelse(
    ((tbal_vector-hysteresis_vector) >= df[,temperature_column]) & (seasonalities_wdep_vector==T | df$holidays==T),
    1, 0)
  if(!is.null(daypart_column)){
    for(s_ in unique(as.character(df[,daypart_column]))){
      if(sum(df[df[,daypart_column]==s_,paste0(temperature_column,"_h")],na.rm=T) <= 5){
        df[df[,daypart_column]==s_, paste0(temperature_column,"_lp_h")] <- 0
        df[df[,daypart_column]==s_, paste0(temperature_column,"_h")] <- 0
        df[df[,daypart_column]==s_, paste0(temperature_column,"_status_h")] <- 0
      }
    }
  } else {
    if(sum(df[,paste0(temperature_column,"_h")],na.rm=T) <= 100){
      df[, paste0(temperature_column,"_lp_h")] <- 0
      df[, paste0(temperature_column,"_h")] <- 0
      df[, paste0(temperature_column,"_status_h")] <- 0
    }
  }
  df[,paste0(temperature_column,"_c")] <- ifelse(
    ((tbal_vector+hysteresis_vector) <= df[,temperature_column]) & (seasonalities_wdep_vector==T | df$holidays==T),
    df[,temperature_column] - (tbal_vector+hysteresis_vector), 0)
  df[,paste0(temperature_column,"_lp_c")] <- ifelse((tbal_vector+hysteresis_vector) <= df[,temperature_column] & (seasonalities_wdep_vector==T | df$holidays==T), 
                                                    lp_temperature_c, 0)
  df[,paste0(temperature_column,"_status_c")] <- ifelse(
    ((tbal_vector+hysteresis_vector) <= df[,temperature_column]) & (seasonalities_wdep_vector==T | df$holidays==T),
    1, 0)
  if(!is.null(daypart_column)){
    for(s_ in unique(as.character(df[,daypart_column]))){
      if(sum(df[df[,daypart_column]==s_,paste0(temperature_column,"_c")],na.rm=T) <= 5){
        df[df[,daypart_column]==s_, paste0(temperature_column,"_lp_c")] <- 0
        df[df[,daypart_column]==s_, paste0(temperature_column,"_c")] <- 0
        df[df[,daypart_column]==s_, paste0(temperature_column,"_status_c")] <- 0
      }
    }
  } else {
    if(sum(df[,paste0(temperature_column,"_c")],na.rm=T) <= 100){
      df[, paste0(temperature_column,"_lp_c")] <- 0
      df[, paste0(temperature_column,"_c")] <- 0
      df[, paste0(temperature_column,"_status_c")] <- 0
    }
  }
  
  return(df)
}


add_cooling_heating_daily_GHI_features <- function(df, temperature_column="temperature", GHI_column="GHI"){
  
  df[,GHI_column] <- ifelse(df[,GHI_column]<10, 0, df[,GHI_column])
  df[,paste0(GHI_column,"_h")] <- ifelse(df[,paste0(temperature_column,"_h")]>0, -df[,GHI_column], 0)
  df[,paste0(GHI_column,"_c")] <- ifelse(df[,paste0(temperature_column,"_c")]>0, df[,GHI_column], 0)
  # df[,paste0(GHI_column,"_h")] <- ifelse(df[,paste0(GHI_column,"_h")]>200, df[,GHI_column], 0)
  # df[,paste0(GHI_column,"_c")] <- ifelse(df[,paste0(GHI_column,"_c")]>200, df[,GHI_column], 0)
  
  return(df)
}

add_cooling_heating_daily_windSpeed_features <- function(df, temperature_column="temperature", windSpeed_column="windSpeed"){
  
  df[,windSpeed_column] <- ifelse(is.na(df[,windSpeed_column]),0,df[,windSpeed_column])
  df[,paste0(windSpeed_column,"_h")] <- ifelse(df[,paste0(temperature_column,"_h")]>0, 
                                               df[,windSpeed_column] * df[,paste0(temperature_column,"_h")], 0)
  df[,paste0(windSpeed_column,"_c")] <- ifelse(df[,paste0(temperature_column,"_c")]>0, 
                                               df[,windSpeed_column] * df[,paste0(temperature_column,"_c")], 0)
  df[,paste0(windSpeed_column)] <- ifelse(df[,paste0(windSpeed_column,"_c")]>0, 
                                          df[,paste0(windSpeed_column,"_c")], df[,paste0(windSpeed_column,"_h")])
  return(df)
}

add_fs_daypart <- function(df, daypart_column, hours_of_each_daypart){
  df$hour_daypart <- ((as.numeric(as.character(df[,daypart_column]))*hours_of_each_daypart)+hours_of_each_daypart/2)
  df_fs <- do.call(cbind,
                   fs(df$hour_daypart/24,#ifelse(df$hour_daypart>=12,(df$hour_daypart/24)-0.5,(df$hour_daypart/24)+0.5),#
                      nharmonics = 3, odd = F, prefix="daypart_fs_"))
  df <- cbind(df,df_fs) 
}

add_fs_weekpart <- function(df, daypart_column="daypart", weekday_column="weekday", hours_of_each_daypart, weekhour_column=NULL){
  if(!is.null(weekhour_column)){
    df$weekhour <- df[,weekhour_column]
  } else {
    df$weekhour <- (as.numeric(df[,weekday_column])-1)*24 +
      ((as.numeric(as.character(df[,daypart_column]))*hours_of_each_daypart)+hours_of_each_daypart/2)
  }
  df_fs <- do.call(cbind,
                   fs(df$weekhour/168,#ifelse(df$weekhour>=(168/2),(df$weekhour/168)-0.5,(df$weekhour/168)+0.5),
                      nharmonics = 3, odd = F, prefix=paste0("weekpart_fs_")))
  df <- cbind(df,df_fs)
}

yhour <- function(time) {
  (yday(time) - 1) * 24 + hour(time)
}

add_fs_daypart_holidays <- function(df, daypart_column, hours_of_each_daypart){
  df$hour_daypart <- ((as.numeric(as.character(df[,daypart_column]))*hours_of_each_daypart)+hours_of_each_daypart/2)
  df_fs <- do.call(cbind,
                   fs(df$hour_daypart/24,#ifelse(df$hour_daypart>=12,(df$hour_daypart/24)-0.5,(df$hour_daypart/24)+0.5),#
                      nharmonics = 2, odd = F, prefix="daypart_holidays_fs_"))*df$holidays
  df <- cbind(df,df_fs) 
}

add_fs_yearpart_holidays <- function(df, time_column="time", yearhour_column=NULL){
  if(!is.null(yearhour_column)){
    df$yearhour <- df[,yearhour_column]
  } else {
    df$yearhour <- yhour(df[,time_column])
  }
  df_fs <- do.call(cbind,
                   fs(df$yearhour/(24*365),
                      nharmonics = 2, odd = F, prefix=paste0("yearpart_holidays_fs_"))) * df$holidays
  df <- cbind(df,df_fs)
}

fs <- function(X, nharmonics, odd=F, prefix="") {
  do.call("c", lapply(1:nharmonics, function(i) {
    if(odd==F){
      val <- list(sin(i * X * 2 * pi), cos(i * X * 2 * pi))
      names(val) <- paste0(prefix,c("sin_", "cos_"), i)
    } else {
      val <- list(sin(i * X * 2 * pi))
      names(val) <- paste0(prefix,c("sin_"), i)
    }
    return(val)
  }))
}

indicators_estimator <- function(characterization, meteo_df, area=1){
  
  tz_local = characterization$config_aggregator$tz_local
  time_column = characterization$config_aggregator$time_column
  value_column = characterization$config_aggregator$value_column
  temperature_column = characterization$config_aggregator$temperature_column
  GHI_column = characterization$config_aggregator$GHI_column
  intercept_column = characterization$config_aggregator$intercept_column
  windSpeed_column = characterization$config_aggregator$windSpeed_column
  group_column = characterization$config_aggregator$group_column
  hours_of_each_daypart = characterization$config_aggregator$hours_of_each_daypart
  
  tryCatch({
    df_s <- aggregator(meteo_df, value_column, temperature_column, windSpeed_column, GHI_column, time_column, intercept_column, tz_local, 
                       group_column, hours_of_each_daypart)
  }, error= function(e){
    return(NULL)
  })
  
  if(class(df_s)=="data.frame"){
    grs <- names(characterization$mod)[mapply(function(i)!identical(characterization$mod[[i]],list()),names(characterization$mod))]
  } else {
    grs <- names(df_s)
  }
  
  # Run the trainer of the model for a one-year-window rolling dataset by months
  results_all <- mclapply(grs, function(gr){
    #gr <- grs[1]
    
    if(class(df_s)=="data.frame"){
      daily_df <- df_s
      daily_df <- merge(daily_df,
                        characterization$classification[characterization$classification$tariff==gr,] %>% dplyr::select(-tariff),
                        by.x="day",by.y="date",all.x=F)
    } else {
      daily_df <- df_s[[gr]]
    }
    
    mod_roll_years <- names(characterization$mod[[gr]])
    
    if(nrow(daily_df)==0){return(NULL)}
    daily_df_roll_years <- rolling_dataset_by_n_months(df = daily_df, date_column = "day", n = 12)[mod_roll_years]
    daily_df_roll_years <- daily_df_roll_years[mapply(function(i){!is.null(daily_df_roll_years[[i]])},1:length(daily_df_roll_years))]
    
    # Total consumptions
    consumptions <- do.call(rbind,lapply(
      names(daily_df_roll_years),
      function(y){
        tryCatch({
          # y=names(daily_df_roll_years)[6]
          roll_year <- unlist(strsplit(y,split = "_"))
          mod <- characterization$mod[[gr]][[y]]
          df_total <- tune_with_params(
            params = c(mod@nuisance$tbal,mod@nuisance$hysteresis,mod@nuisance$seasonalities_wdep,mod@nuisance$thermal_time_constant,mod@nuisance$training_without_holidays),
            df = daily_df_roll_years[[y]],
            temperature_column = temperature_column,
            GHI_column = GHI_column,
            windSpeed_column = windSpeed_column,
            daypart_column = "daypart",
            weekday_column = intercept_column,
            time_column = time_column,
            hours_of_each_daypart = hours_of_each_daypart,
            training_seasonality_levels = mod@nuisance$seasonality_levels
          )
          df_total_without_covid <- tune_with_params(
            params = c(mod@nuisance$tbal,mod@nuisance$hysteresis,mod@nuisance$seasonalities_wdep,mod@nuisance$thermal_time_constant,mod@nuisance$training_without_holidays),
            df = daily_df_roll_years[[y]],
            temperature_column = temperature_column,
            GHI_column = GHI_column,
            windSpeed_column = windSpeed_column,
            daypart_column = "daypart",
            weekday_column = intercept_column,
            time_column = time_column,
            hours_of_each_daypart = hours_of_each_daypart,
            training_seasonality_levels = mod@nuisance$seasonality_levels,
            force_covid_lockdown=F
          )
          df_no_weather_dep <- tune_with_params(
            params = c(mod@nuisance$tbal,mod@nuisance$hysteresis,mod@nuisance$seasonalities_wdep,mod@nuisance$thermal_time_constant,mod@nuisance$training_without_holidays),
            df = daily_df_roll_years[[y]],
            temperature_column = temperature_column,
            GHI_column = GHI_column,
            windSpeed_column = windSpeed_column,
            force_weather_heating_vars_to_0 = T,
            force_weather_cooling_vars_to_0 = T,
            daypart_column = "daypart",
            weekday_column = intercept_column,
            time_column = time_column,
            hours_of_each_daypart = hours_of_each_daypart,
            training_seasonality_levels = mod@nuisance$seasonality_levels
          )
          df_weather_dep_heating <- tune_with_params(
            params = c(mod@nuisance$tbal,mod@nuisance$hysteresis,mod@nuisance$seasonalities_wdep,mod@nuisance$thermal_time_constant,mod@nuisance$training_without_holidays),
            df = daily_df_roll_years[[y]],
            temperature_column = temperature_column,
            GHI_column = GHI_column,
            windSpeed_column = windSpeed_column,
            force_weather_heating_vars_to_0 = F,
            force_weather_cooling_vars_to_0 = T,
            daypart_column = "daypart",
            weekday_column = intercept_column,
            time_column = time_column,
            hours_of_each_daypart = hours_of_each_daypart,
            training_seasonality_levels = mod@nuisance$seasonality_levels
          )
          df_weather_dep_cooling <- tune_with_params(
            params = c(mod@nuisance$tbal,mod@nuisance$hysteresis,mod@nuisance$seasonalities_wdep,mod@nuisance$thermal_time_constant,mod@nuisance$training_without_holidays),
            df = daily_df_roll_years[[y]],
            temperature_column = temperature_column,
            GHI_column = GHI_column,
            windSpeed_column = windSpeed_column,
            force_weather_heating_vars_to_0 = T,
            force_weather_cooling_vars_to_0 = F,
            daypart_column = "daypart",
            weekday_column = intercept_column,
            time_column = time_column,
            hours_of_each_daypart = hours_of_each_daypart,
            training_seasonality_levels = mod@nuisance$seasonality_levels
          )
          total <- #(predict(mod,(as.matrix(model.matrix(mod@nuisance$formula[-2],df_total))))[,1])
            c((as.matrix(model.matrix(mod@nuisance$formula[-2],df_total))[,names(penalized::coefficients(mod))]) %*% penalized::coefficients(mod))
          total_without_covid <- c((as.matrix(model.matrix(mod@nuisance$formula[-2],df_total_without_covid))[,names(penalized::coefficients(mod))]) %*% penalized::coefficients(mod))
          no_weather_dep <- #(predict(mod,as.matrix(model.matrix(mod@nuisance$formula[-2],df_no_weather_dep)))[,1])
            c((as.matrix(model.matrix(mod@nuisance$formula[-2],df_no_weather_dep))[,names(penalized::coefficients(mod))]) %*% penalized::coefficients(mod))
          weather_dep_heating <- #(predict(mod,as.matrix(model.matrix(mod@nuisance$formula[-2],df_weather_dep_heating)))[,1])
            c((as.matrix(model.matrix(mod@nuisance$formula[-2],df_weather_dep_heating))[,names(penalized::coefficients(mod))]) %*% penalized::coefficients(mod)) - no_weather_dep
          
          temp_lp <- grepl("temperature_lp",names(penalized::coefficients(mod)))
          temp <- grepl("temperature_h|temperature_c",names(penalized::coefficients(mod)))
          wind <- grepl("windSpeed_c|windSpeed_h",names(penalized::coefficients(mod)))
          
          temp_lp_heating <- c(as.matrix(model.matrix(mod@nuisance$formula[-2],df_weather_dep_heating))[,names(penalized::coefficients(mod))[temp_lp]] %*% penalized::coefficients(mod)[temp_lp])
          temp_heating <- c(as.matrix(model.matrix(mod@nuisance$formula[-2],df_weather_dep_heating))[,names(penalized::coefficients(mod))[temp]] %*% penalized::coefficients(mod)[temp])
          wind_heating <- c(as.matrix(model.matrix(mod@nuisance$formula[-2],df_weather_dep_heating))[,names(penalized::coefficients(mod))[wind]] %*% penalized::coefficients(mod)[wind])
          
          #weather_dep_heating <- ifelse(weather_dep_heating>0,weather_dep_heating,0)
          weather_dep_cooling <- 
            c((as.matrix(model.matrix(mod@nuisance$formula[-2],df_weather_dep_cooling))[,names(penalized::coefficients(mod))]) %*% penalized::coefficients(mod)) - no_weather_dep
          
          temp_lp_cooling <- c(as.matrix(model.matrix(mod@nuisance$formula[-2],df_weather_dep_cooling))[,names(penalized::coefficients(mod))[temp_lp]] %*% penalized::coefficients(mod)[temp_lp])
          temp_cooling <- c(as.matrix(model.matrix(mod@nuisance$formula[-2],df_weather_dep_cooling))[,names(penalized::coefficients(mod))[temp]] %*% penalized::coefficients(mod)[temp])
          wind_cooling <- c(as.matrix(model.matrix(mod@nuisance$formula[-2],df_weather_dep_cooling))[,names(penalized::coefficients(mod))[wind]] %*% penalized::coefficients(mod)[wind])
          
          holidays_component <- merge(df_total,mod@nuisance$model[,c("time","holidays_component")],by = "time",all.x = T)$holidays_component
          total <- total + ifelse(is.finite(holidays_component),holidays_component,0)
          total_without_covid <- total_without_covid + ifelse(is.finite(holidays_component),holidays_component,0)
          covid_affectance <- total - total_without_covid
          #weather_dep_cooling <- ifelse(weather_dep_cooling>0,weather_dep_cooling,0)
          #no_weather_dep <- total - weather_dep_cooling - weather_dep_heating
          df_result <- data.frame(
            "monthDay"=strftime(df_total[,time_column],"%m-%d"),
            "dayHour"=as.numeric(as.character(df_total[,"daypart"]))*hours_of_each_daypart ,
            "total"=total, 
            "baseload"=no_weather_dep,
            "holidays"=holidays_component,
            "heating"=weather_dep_heating, 
            "heatingEnvelope"=temp_lp_heating,
            "heatingVentilation"=temp_heating,
            "heatingAirInfiltration"=wind_heating,
            "cooling"=weather_dep_cooling,
            "coolingEnvelope"=temp_lp_cooling,
            "coolingVentilation"=temp_cooling,
            "coolingAirInfiltration"=wind_cooling,
            "totalDuringCovidLockdown"=ifelse(
              as.Date(df_total[,time_column]) >= as.Date("2020-03-15") &  
                as.Date(df_total[,time_column]) <= as.Date("2020-06-21"),
              total_without_covid,
              0),
            "covid"=covid_affectance,
            "month"=month(df_total[,time_column]),
            "rollYear"=y,
            "iniDate"=if(y=="last_period"){NA}else{as.Date(paste0(roll_year[1],"01"),"%Y%m%d")},
            "endDate"=if(y=="last_period"){NA}else{as.Date(paste0(roll_year[2],"01"),"%Y%m%d")+months(1)-days(1)}
          )
          df_result[order(df_result$monthDay),]
        },error=function(e)NULL)
      }
    ))
    
    # Indicators
    weather_indicators <- do.call(rbind,lapply(
      X=names(daily_df_roll_years),
      function(y){
        roll_year <- unlist(strsplit(y,split = "_"))
        mod <- characterization$mod[[gr]][[y]]
        indicator_names <- colnames(mod@nuisance$model)[grepl("_h$|_c$",colnames(mod@nuisance$model))]
        indicator_names <- indicator_names[mapply(function(i)grepl(i,as.character(mod@nuisance$formula[-2])[2]),indicator_names)]
        do.call(rbind, lapply(FUN=function(indicator_name){
          if(any(grepl("weekpart",colnames(mod@nuisance$model))) & !any(grepl("^s$",colnames(mod@nuisance$model)))){
            df_test <- mod@nuisance$model[1,]
            df_test <- data.frame(as.data.frame(df_test[!grepl("^weekpart|^daypart|^yearpart",colnames(mod@nuisance$model))]),
                                  "daypart"=(1:(24/hours_of_each_daypart))-1,
                                  "weekpart"=(1:(168/hours_of_each_daypart))-1)
            df_test$weekhour <- df_test$weekpart * hours_of_each_daypart
            df_test <- add_fs_weekpart(df=df_test,hours_of_each_daypart = hours_of_each_daypart,
                                       weekhour_column = "weekhour")
            df_test <- add_fs_daypart(df=df_test,daypart_column = "daypart", hours_of_each_daypart = hours_of_each_daypart)
            df_test <- do.call(rbind, lapply( seq(24*15,168*48,24*30)-1, #Monthly from day 15.
                                              function(x){cbind(df_test,"yearhour"=x)}
            ))
            df_test <- add_fs_yearpart(df=df_test,yearhour_column = "yearhour")
            df_test$s <- "01"
            # df_test$daypart <- as.factor(df_test$daypart)
          } else if(any(grepl("^s$",colnames(mod@nuisance$model)))){
            df_test <- mod@nuisance$model[1,]
            df_test <- data.frame(as.data.frame(df_test[!grepl("^s$|^daypart|^weekpart",colnames(mod@nuisance$model))]),
                                  "daypart"=(1:(24/hours_of_each_daypart))-1,
                                  "weekpart"=(1:(168/hours_of_each_daypart))-1)
            df_test$weekhour <- df_test$weekpart * hours_of_each_daypart
            df_test <- add_fs_weekpart(df=df_test,hours_of_each_daypart = hours_of_each_daypart,
                                       weekhour_column = "weekhour")
            df_test <- add_fs_daypart(df=df_test,daypart_column = "daypart", hours_of_each_daypart = hours_of_each_daypart)
            df_test <- do.call(rbind, lapply( sort(unique(mod@nuisance$model$s)),#gsub("s","",colnames(mod@nuisance$model)[grepl("^s[0-9]{2}$",colnames(mod@nuisance$model))])),
                                              function(x){cbind(df_test,"s"=x)}
            ))
            df_test$yearhour <- 0
          } else {
            df_test <- mod@nuisance$model[1,]
            df_test$daypart <- 0
            df_test$weekpart <- 0
            df_test$weekhour <- 0
            df_test$yearhour <- 0
            df_test$s <- "01"
          }
          df_test$daypart <- as.factor(df_test$daypart)
          df_test$weekpart <- as.factor(df_test$weekpart)
          df_test[,grepl("_h|_c",colnames(df_test))] <- 0
          no_weather <- (predict(mod, as.matrix(model.matrix(mod@nuisance$formula[-2],df_test)))[,1])
          df_test[,indicator_name] <- 1
          # if(grepl("^temperature",indicator_name)==T){
          #   df_test[,gsub("_","_status_",indicator_name)] <- 1
          # }
          indicator_value <- (predict(mod, as.matrix(model.matrix(mod@nuisance$formula[-2],df_test)))[,1] - no_weather)
          df_result <- data.frame(
            "indicator" = indicator_value,
            "indicatorName" = gsub("_c$|_h$","",indicator_name),
            "mode" = if(grepl("_h$",indicator_name)) "heating" else if(grepl("_c$",indicator_name)) "cooling" else "all",
            "dayPart" = df_test$daypart,
            "weekPart" = df_test$weekpart,
            "weekHour" = df_test$weekhour,
            "yearHour" = df_test$yearhour,
            "s" = df_test$s,
            "month" = strftime(as.POSIXct("2020-01-01 00:00:00")+(df_test$yearhour)*3600,format="%b"),
            "rollYear" = y,
            "iniDate" = if(y=="last_period"){NA}else{as.Date(paste0(roll_year[1],"01"),"%Y%m%d")},
            "endDate" = if(y=="last_period"){NA}else{as.Date(paste0(roll_year[2],"01"),"%Y%m%d") + months(1) - days(1)}
          )
        }, indicator_names))
      }
    ))
    
    tbal_indicators <- do.call(rbind,lapply(
      X=names(daily_df_roll_years),
      function(y){
        roll_year <- unlist(strsplit(y,split = "_"))
        mod <- characterization$mod[[gr]][[y]]
        do.call(rbind, lapply(FUN=function(mode_){
          if(hours_of_each_daypart<24 & !any(grepl("^s$",colnames(mod@nuisance$model)))){
            df_test <- data.frame(mod@nuisance$model[1,],
                                  "daypart"=1:(24/hours_of_each_daypart)-1,
                                  "weekpart"=0:((168/hours_of_each_daypart)-1))
            df_test$weekhour <- df_test$weekpart * hours_of_each_daypart
            df_test <- do.call(rbind, lapply( seq(1,168*52,24*30)-1,
                                              function(x){cbind(df_test,"yearhour"=x)}
            ))
            df_test$s <- "01"
            # df_test$daypart <- as.factor(df_test$daypart)
          } else if(any(grepl("^s$",colnames(mod@nuisance$model)))){
            df_test <- mod@nuisance$model[1,]
            df_test <- data.frame(as.data.frame(df_test[!grepl("^s$|^daypart|^weekpart",colnames(mod@nuisance$model))]),
                                  "daypart"=(1:(24/hours_of_each_daypart))-1,
                                  "weekpart"=(1:(168/hours_of_each_daypart))-1)
            df_test$weekhour <- df_test$weekpart * hours_of_each_daypart
            df_test <- add_fs_weekpart(df=df_test,hours_of_each_daypart = hours_of_each_daypart,
                                       weekhour_column = "weekhour")
            df_test <- add_fs_daypart(df=df_test,daypart_column = "daypart", hours_of_each_daypart = hours_of_each_daypart)
            df_test <- do.call(rbind, lapply( sort(unique(mod@nuisance$model$s)),#gsub("s","",colnames(mod@nuisance$model)[grepl("^s[0-9]{2}$",colnames(mod@nuisance$model))])),
                                              function(x){cbind(df_test,"s"=x)}
            ))
            df_test$yearhour <- 0
            # df_test$daypart <- as.factor(df_test$daypart)
          } else {
            df_test <- mod@nuisance$model[1,]
            df_test$daypart <- 0
            df_test$weekpart <- 0
            df_test$weekhour <- 0
            df_test$yearhour <- 0
            df_test$s <- "01"
          }
          df_test$daypart <- as.factor(df_test$daypart)
          df_test$weekpart <- as.factor(df_test$weekpart)
          e <- (as.integer(as.character(df_test$daypart))+1)#+max(as.numeric(as.character(df_test$daypart))+1)*(as.numeric(as.character(df_test$s))-1)
          df_result <- data.frame(
            "indicator" = if(mode_=="heating"){
              mod@nuisance$tbal[e] - mod@nuisance$hysteresis[e]
            } else {
              mod@nuisance$tbal[e] + mod@nuisance$hysteresis[e]
            },
            "indicatorName" = "balancePoint",
            "mode" = mode_,
            "dayPart" = df_test$daypart,
            "weekPart" = df_test$weekpart,
            "weekHour" = df_test$weekhour,
            "yearHour" = df_test$yearhour,
            "s" = df_test$s,
            "month" = strftime(as.POSIXct("2020-01-01 00:00:00")+(df_test$yearhour)*3600,format="%b"),
            "rollYear" = y,
            "iniDate" = if(y=="last_period"){NA}else{as.Date(paste0(roll_year[1],"01"),"%Y%m%d")},
            "endDate" = if(y=="last_period"){NA}else{as.Date(paste0(roll_year[2],"01"),"%Y%m%d") + months(1) - days(1)}
          )
          df_result$indicator <- ifelse(mod@nuisance$seasonalities_wdep[as.numeric(as.character(df_result$s))]==1,df_result$indicator,NA)
          df_result
        }, c("heating","cooling")))
      }
    ))
    
    time_constant_indicators <- do.call(rbind,lapply(
      X=names(daily_df_roll_years),
      function(y){
        roll_year <- unlist(strsplit(y,split = "_"))
        mod <- characterization$mod[[gr]][[y]]
        do.call(rbind, lapply(FUN=function(mode_){
          if(hours_of_each_daypart<24 & !any(grepl("^s$",colnames(mod@nuisance$model)))){
            df_test <- data.frame(mod@nuisance$model[1,],
                                  "daypart"=1:(24/hours_of_each_daypart)-1,
                                  "weekpart"=0:((168/hours_of_each_daypart)-1))
            df_test$weekhour <- df_test$weekpart * hours_of_each_daypart
            df_test <- do.call(rbind, lapply( seq(1,168*52,24*30)-1,
                                              function(x){cbind(df_test,"yearhour"=x)}
            ))
            df_test$s <- "01"
            # df_test$daypart <- as.factor(df_test$daypart)
          } else if(any(grepl("^s$",colnames(mod@nuisance$model)))){
            df_test <- mod@nuisance$model[1,]
            df_test <- data.frame(as.data.frame(df_test[!grepl("^s$|^daypart|^weekpart",colnames(mod@nuisance$model))]),
                                  "daypart"=(1:(24/hours_of_each_daypart))-1,
                                  "weekpart"=(1:(168/hours_of_each_daypart))-1)
            df_test$weekhour <- df_test$weekpart * hours_of_each_daypart
            df_test <- add_fs_weekpart(df=df_test,hours_of_each_daypart = hours_of_each_daypart,
                                       weekhour_column = "weekhour")
            df_test <- add_fs_daypart(df=df_test,daypart_column = "daypart", hours_of_each_daypart = hours_of_each_daypart)
            df_test <- do.call(rbind, lapply( sort(unique(mod@nuisance$model$s)),#gsub("s","",colnames(mod@nuisance$model)[grepl("^s[0-9]{2}$",colnames(mod@nuisance$model))])),
                                              function(x){cbind(df_test,"s"=x)}
            ))
            df_test$yearhour <- 0
            # df_test$daypart <- as.factor(df_test$daypart)
          } else {
            df_test <- mod@nuisance$model[1,]
            df_test$daypart <- 0
            df_test$weekpart <- 0
            df_test$weekhour <- 0
            df_test$yearhour <- 0
            df_test$s <- "01"
          }
          df_test$daypart <- as.factor(df_test$daypart)
          df_test$weekpart <- as.factor(df_test$weekpart)
          #e <- (as.integer(as.character(df_test$daypart))+1)#+max(as.numeric(as.character(df_test$daypart))+1)*(as.numeric(as.character(df_test$s))-1)
          df_result <- data.frame(
            "indicator" = if(mode_=="heating"){
              mod@nuisance$thermal_time_constant# * mod@nuisance$seasonalities_wdep[e]
            } else {
              mod@nuisance$thermal_time_constant# * mod@nuisance$seasonalities_wdep[e]
            },
            "indicatorName" = "thermalTimeConstant",
            "mode" = mode_,
            "dayPart" = df_test$daypart,
            "weekPart" = df_test$weekpart,
            "weekHour" = df_test$weekhour,
            "yearHour" = df_test$yearhour,
            "s" = df_test$s,
            "month" = strftime(as.POSIXct("2020-01-01 00:00:00")+(df_test$yearhour)*3600,format="%b"),
            "rollYear" = y,
            "iniDate" = if(y=="last_period"){NA}else{as.Date(paste0(roll_year[1],"01"),"%Y%m%d")},
            "endDate" = if(y=="last_period"){NA}else{as.Date(paste0(roll_year[2],"01"),"%Y%m%d") + months(1) - days(1)}
          )
          df_result$indicator <- ifelse(mod@nuisance$seasonalities_wdep[as.numeric(as.character(df_result$s))]==1,df_result$indicator,NA)
          df_result
        }, c("heating","cooling")))
      }
    ))
    
    weather_indicators <- rbind(weather_indicators, tbal_indicators, time_constant_indicators)
    
    baseload_indicators <- do.call(rbind,lapply(
      X=names(daily_df_roll_years),
      function(y){
        roll_year <- unlist(strsplit(y,split = "_"))
        mod <- characterization$mod[[gr]][[y]]
        if(any(grepl("weekpart",colnames(mod@nuisance$model))) & !any(grepl("^s$",colnames(mod@nuisance$model)))){
          df_test <- mod@nuisance$model[1,]
          df_test <- data.frame(as.data.frame(df_test[!grepl("^weekpart|^daypart|^yearpart",colnames(mod@nuisance$model))]),
                                "daypart"=(1:(24/hours_of_each_daypart))-1,
                                "weekpart"=(1:(168/hours_of_each_daypart))-1)
          df_test$weekhour <- df_test$weekpart * hours_of_each_daypart
          df_test <- add_fs_weekpart(df=df_test,hours_of_each_daypart = hours_of_each_daypart,
                                     weekhour_column = "weekhour")
          df_test <- add_fs_daypart(df=df_test,daypart_column = "daypart", hours_of_each_daypart = hours_of_each_daypart)
          df_test <- do.call(rbind, lapply( seq(24*15,168*52,24*30)-1, #Monthly from day 15.
                                            function(x){cbind(df_test,"yearhour"=x)}
          ))
          df_test <- add_fs_yearpart(df=df_test,yearhour_column = "yearhour")
          df_test$s <- "01"
          # df_test$daypart <- as.factor(df_test$daypart)
        } else if(any(grepl("^s$",colnames(mod@nuisance$model)))){
          df_test <- mod@nuisance$model[1,]
          df_test <- data.frame(as.data.frame(df_test[!grepl("^s$|^daypart|^weekpart",colnames(mod@nuisance$model))]),
                                "daypart"=(1:(24/hours_of_each_daypart))-1,
                                "weekpart"=(1:(168/hours_of_each_daypart))-1)
          df_test$weekhour <- df_test$weekpart * hours_of_each_daypart
          df_test <- add_fs_weekpart(df=df_test,hours_of_each_daypart = hours_of_each_daypart,
                                     weekhour_column = "weekhour")
          df_test <- add_fs_daypart(df=df_test,daypart_column = "daypart", hours_of_each_daypart = hours_of_each_daypart)
          df_test <- do.call(rbind, lapply( sort(unique(mod@nuisance$model$s)),#gsub("s","",colnames(mod@nuisance$model)[grepl("^s[0-9]{2}$",colnames(mod@nuisance$model))])),
                                            function(x){cbind(df_test,"s"=x)}
          ))
          df_test$yearhour <- 0
          # df_test$daypart <- as.factor(df_test$daypart)
          # df_test$daypart <- as.factor(df_test$daypart)
        } else {
          df_test <- mod@nuisance$model[1,]
          df_test$daypart <- 0 
          df_test$weekpart <- 0
          df_test$weekhour <- 0
          df_test$yearhour <- 0
        }
        df_test$daypart <- as.factor(df_test$daypart)
        df_test$weekpart <- as.factor(df_test$weekpart)
        df_test[,grepl("windSpeed|_h|_c",colnames(df_test))] <- 0
        no_weather <- (predict(mod, as.matrix(model.matrix(mod@nuisance$formula[-2],df_test)))[,1])
        df_result <- data.frame(
          "indicator" = no_weather,
          "indicatorName" = df_test$weekpart,
          "mode" = "always",
          "dayPart" = df_test$daypart,
          "weekPart" = df_test$weekpart,
          "weekHour" = df_test$weekhour,
          "yearHour" = df_test$yearhour,
          "s" = df_test$s,
          "month" = strftime(as.POSIXct("2020-01-01 00:00:00")+(df_test$yearhour)*3600,format="%b"),
          "rollYear" = y,
          "iniDate" = if(y=="last_period"){NA}else{as.Date(paste0(roll_year[1],"01"),"%Y%m%d")},
          "endDate" = if(y=="last_period"){NA}else{as.Date(paste0(roll_year[2],"01"),"%Y%m%d") + months(1) - days(1)}
        )
      }
    ))
    
    # ggplot(consumptions) + geom_line(aes(as.Date(paste0("2020-",monthday)),total)) + geom_line(aes(as.Date(paste0("2020-",monthday)),no_weather_dep),col="green") +
    #   geom_line(aes(as.Date(paste0("2020-",monthday)),weather_dep_heating),col="blue") + geom_line(aes(as.Date(paste0("2020-",monthday)),weather_dep_cooling),col="red") +
    #   facet_wrap(~end_date,ncol=1) + scale_x_date(date_labels = "%b")
    # ggplot(weather_indicators) + geom_line(aes(end_date,indicator, col=daypart)) + facet_grid(indicator_name~mode, scales = "free_y")
    # ggplot(baseload_indicators) + geom_line(
    #   aes(as.numeric(as.character(factor(indicator_name,labels=0:6)))*24 + as.numeric(daypart)*characterization$config_aggregator$hours_of_each_daypart,indicator)) +
    #   facet_wrap(~end_date,ncol=1)
    
    return(list(
      "consumption_results"= consumptions, 
      "weather_dep_indicators" = weather_indicators,
      "baseload_indicators" = baseload_indicators)
    )
    
  })
  
  names(results_all) <- grs
  check_empty <- mapply(function(i)!is.null(results_all[[i]]),grs)
  results_all <- results_all[check_empty]
  grs <- grs[check_empty]
  consumptions <- do.call(rbind,lapply(grs, function(i){cbind(results_all[[i]]$consumption_results,"group"=i)}))
  weather_indicators <- do.call(rbind,lapply(grs, function(i){cbind(results_all[[i]]$weather_dep_indicators,"group"=i)}))
  baseload_indicators <- do.call(rbind,lapply(grs, function(i){cbind(results_all[[i]]$baseload_indicators,"group"=i)}))
  colnames(consumptions)[colnames(consumptions)=="group"] <- group_column
  colnames(weather_indicators)[colnames(weather_indicators)=="group"] <- group_column
  colnames(baseload_indicators)[colnames(baseload_indicators)=="group"] <- group_column
  
  weather_indicators$indicatorNameUnit <- factor(weather_indicators$indicatorName, 
                                                 c("temperature","temperature_lp","balancePoint",
                                                   "windSpeed","GHI","thermalTimeConstant"),
                                                 c('"U"[raw]*"[W/m"^2*"K]"','"U"[lp]*"[W/m"^2*"K]"','T^"bal"*"["*degree*"C]"',
                                                   'I^"air"*"[Ws/m"^3*"K]"','I^"sol"*"[m"^"4"*"]"','tau*"[h]"'))
  
  # Consumption_summary_per_year
  consumptions$agg <- paste(consumptions[,group_column],consumptions$rollYear,sep="~")
  consumptions_summary <- smartAgg(consumptions, "agg", 
                                   function(x){mean(x,na.rm=T)*8760/1000}, # W/m2 -> kWh/m2year
                                   c("total", "baseload", "heating", "cooling", "holidays","totalDuringCovidLockdown","covid",
                                     "coolingEnvelope","coolingVentilation","coolingAirInfiltration",
                                     "heatingEnvelope","heatingVentilation","heatingAirInfiltration"),
                                   function(x){x[1]}, 
                                   c("tariff", "rollYear"),
                                   
                                   catN=F)
  consumptions_summary_m2 <- consumptions_summary[,c("total", "baseload", "heating", "cooling","holidays","totalDuringCovidLockdown","covid",
                                                     "coolingEnvelope","coolingVentilation","coolingAirInfiltration",
                                                     "heatingEnvelope","heatingVentilation","heatingAirInfiltration")]
  colnames(consumptions_summary_m2) <- paste0(colnames(consumptions_summary_m2),"PerArea")
  consumptions_summary[,c("total", "baseload", "heating", "cooling","holidays","totalDuringCovidLockdown","covid",
                          "coolingEnvelope","coolingVentilation","coolingAirInfiltration",
                          "heatingEnvelope","heatingVentilation","heatingAirInfiltration")] <- 
    consumptions_summary[,c("total", "baseload", "heating", "cooling","holidays","totalDuringCovidLockdown","covid",
                            "coolingEnvelope","coolingVentilation","coolingAirInfiltration",
                            "heatingEnvelope","heatingVentilation","heatingAirInfiltration")] * area #kWh/m2year -> kWh/year
  roll_year <- do.call(rbind,strsplit(consumptions_summary$rollYear,"_"))
  consumptions_summary <- cbind(consumptions_summary,data.frame(
    "iniDate"=ifelse(consumptions_summary$rollYear=="last_period",NA,as.Date(paste0(roll_year[,1],"01"),"%Y%m%d")),
    "endDate"=ifelse(consumptions_summary$rollYear=="last_period",NA,as.Date(paste0(roll_year[,2],"01"),"%Y%m%d") + months(1) - days(1))
  ))
  consumptions_summary <- cbind(consumptions_summary,mapply(
    c("baseload","heating","cooling","holidays","covid",
      "coolingEnvelope","coolingVentilation","coolingAirInfiltration",
      "heatingEnvelope","heatingVentilation","heatingAirInfiltration"),
    FUN = function(i){
      if (i=="covid"){
        (consumptions_summary[,i]/consumptions_summary[,"totalDuringCovidLockdown"]) * 100
      } else if (i %in% c("coolingEnvelope","coolingVentilation","coolingAirInfiltration")){
        ifelse(consumptions_summary[,"cooling"]>0,
               (consumptions_summary[,i]/consumptions_summary[,"cooling"]) * 100,
               0)
      } else if (i %in% c("heatingEnvelope","heatingVentilation","heatingAirInfiltration")){
        ifelse(consumptions_summary[,"heating"]>0,
               (consumptions_summary[,i]/consumptions_summary[,"heating"]) * 100,
               0)
      } else {
        (consumptions_summary[,i]/consumptions_summary[,"total"])*100
      }
    }
  )
  )
  colnames(consumptions_summary)[
    (length(colnames(consumptions_summary))-10):length(colnames(consumptions_summary))] <- 
    c("baseloadPercentage", "heatingPercentage", "coolingPercentage","holidaysPercentage","covidPercentage",
      "coolingEnvelopePercentage","coolingVentilationPercentage","coolingAirInfiltrationPercentage",
      "heatingEnvelopePercentage","heatingVentilationPercentage","heatingAirInfiltrationPercentage")
  consumptions_summary$iniDate <- as.Date(consumptions_summary$iniDate)
  consumptions_summary$endDate <- as.Date(consumptions_summary$endDate)
  consumptions_summary <- cbind(consumptions_summary, consumptions_summary_m2)
  
  # Consumption summary per natural month and part of day
  consumptions$agg <- paste(consumptions[,group_column],consumptions$rollYear,consumptions$month,consumptions$dayHour,sep="~")
  consumptions_summary_detailed <- smartAgg(consumptions, "agg",
                                            function(x){mean(x,na.rm=T)*720/hours_of_each_daypart/1000}, # W -> kWh/(month*daypart)
                                            c("total", "baseload", "heating", "cooling","holidays","covid","totalDuringCovidLockdown",
                                              "coolingEnvelope","coolingVentilation","coolingAirInfiltration",
                                              "heatingEnvelope","heatingVentilation","heatingAirInfiltration"
                                            ),
                                            function(x){x[1]},
                                            c("tariff", "rollYear", "dayHour", "month"),
                                            catN=F)
  consumptions_summary_detailed_m2 <- consumptions_summary_detailed[,
                                                                    c("total", "baseload", "heating", "cooling","holidays","totalDuringCovidLockdown","covid",
                                                                      "coolingEnvelope","coolingVentilation","coolingAirInfiltration",
                                                                      "heatingEnvelope","heatingVentilation","heatingAirInfiltration")]
  colnames(consumptions_summary_detailed_m2) <- paste0(colnames(consumptions_summary_detailed_m2),"PerArea")
  consumptions_summary_detailed[,c("total", "baseload", "heating", "cooling","holidays","covid","totalDuringCovidLockdown",
                                   "coolingEnvelope","coolingVentilation","coolingAirInfiltration",
                                   "heatingEnvelope","heatingVentilation","heatingAirInfiltration")] <- 
    consumptions_summary_detailed[,c("total", "baseload", "heating", "cooling","holidays","covid","totalDuringCovidLockdown",
                                     "coolingEnvelope","coolingVentilation","coolingAirInfiltration",
                                     "heatingEnvelope","heatingVentilation","heatingAirInfiltration")] * area
  roll_year <- do.call(rbind,strsplit(consumptions_summary_detailed$rollYear,"_"))
  consumptions_summary_detailed <- cbind(consumptions_summary_detailed,data.frame(
    "iniDate"=ifelse(consumptions_summary_detailed$rollYear=="last_period",NA,as.Date(paste0(roll_year[,1],"01"),"%Y%m%d")),
    "endDate"=ifelse(consumptions_summary_detailed$rollYear=="last_period",NA,as.Date(paste0(roll_year[,2],"01"),"%Y%m%d") + months(1) - days(1))
  ))
  consumptions_summary_detailed <- cbind(consumptions_summary_detailed,mapply(
    c("baseload","heating","cooling","holidays","covid",
      "coolingEnvelope","coolingVentilation","coolingAirInfiltration",
      "heatingEnvelope","heatingVentilation","heatingAirInfiltration"),
    FUN = function(i){
      if (i=="covid"){
        (consumptions_summary_detailed[,i]/consumptions_summary_detailed["totalDuringCovidLockdown"])*100
      } else if (i %in% c("coolingEnvelope","coolingVentilation","coolingAirInfiltration")){
        ifelse(consumptions_summary_detailed[,"cooling"]>0,
               (consumptions_summary_detailed[,i]/consumptions_summary_detailed[,"cooling"]) * 100,
               0)
      } else if (i %in% c("heatingEnvelope","heatingVentilation","heatingAirInfiltration")){
        ifelse(consumptions_summary_detailed[,"heating"]>0,
               (consumptions_summary_detailed[,i]/consumptions_summary_detailed[,"heating"]) * 100,
               0)
      } else {
        (consumptions_summary_detailed[,i]/consumptions_summary_detailed["total"])*100
      }
    }
  )
  )
  colnames(consumptions_summary_detailed)[
    (length(colnames(consumptions_summary_detailed))-10):length(colnames(consumptions_summary_detailed))] <- 
    c("baseloadPercentage", "heatingPercentage", "coolingPercentage","holidaysPercentage","covidPercentage",
      "coolingEnvelopePercentage","coolingVentilationPercentage","coolingAirInfiltrationPercentage",
      "heatingEnvelopePercentage","heatingVentilationPercentage","heatingAirInfiltrationPercentage")
  consumptions_summary_detailed$iniDate <- as.Date(consumptions_summary_detailed$iniDate)
  consumptions_summary_detailed$endDate <- as.Date(consumptions_summary_detailed$endDate)
  consumptions_summary_detailed <- cbind(consumptions_summary_detailed, consumptions_summary_detailed_m2)
  
  # Multiply the raw consumption data by the area (W/m² to kWh) and delete the aggregation column
  consumptions$agg <- NULL
  consumptions[,c("total", "baseload", "heating", "cooling","holidays","covid",
                  "coolingEnvelope","coolingVentilation","coolingAirInfiltration",
                  "heatingEnvelope","heatingVentilation","heatingAirInfiltration",
                  "totalDuringCovidLockdown")] <- 
    consumptions[,c("total", "baseload", "heating", "cooling","holidays","covid",
                    "coolingEnvelope","coolingVentilation","coolingAirInfiltration",
                    "heatingEnvelope","heatingVentilation","heatingAirInfiltration",
                    "totalDuringCovidLockdown")] * area * hours_of_each_daypart / 1000
  consumptions$originalArea <- area
  
  # plot(consumptions_summary_detailed$total[consumptions_summary_detailed$tariff=="2.0A" & 
  #                                            is.na(consumptions_summary_detailed$iniDate)],type="l", ylim=c(0,60))
  # lines(consumptions_summary_detailed$baseload[consumptions_summary_detailed$tariff=="2.0A" & 
  #          is.na(consumptions_summary_detailed$iniDate)],col="grey")
  # lines(consumptions_summary_detailed$holidays[consumptions_summary_detailed$tariff=="2.0A" & 
  #                                                is.na(consumptions_summary_detailed$iniDate)],col="yellow")
  # lines(consumptions_summary_detailed$heating[consumptions_summary_detailed$tariff=="2.0A" & 
  #                                                is.na(consumptions_summary_detailed$iniDate)],col="red")
  # lines(consumptions_summary_detailed$cooling[consumptions_summary_detailed$tariff=="2.0A" & 
  #                                                is.na(consumptions_summary_detailed$iniDate)],col="blue")
  
  daypart_values <- data.frame(
    "hour" = 0:23,
    "dayPart" = rep(0:((24/hours_of_each_daypart)-1), each=hours_of_each_daypart)
  )
  daypart_values <- smartAgg(daypart_values,by = "dayPart",function(i)min(i),"hour",function(i)max(i),"hour",catN = F)
  colnames(daypart_values) <- c("dayPart","minH","maxH")
  daypart_values$hourLabel <- paste0(daypart_values$minH,"-",daypart_values$maxH,"h")
  
  return(list(
    "consumption_disaggregation" = consumptions,
    "consumption_disaggregation_summary" = consumptions_summary,
    "consumption_disaggregation_summary_detailed" = consumptions_summary_detailed,
    "weather_dep_indicators" = weather_indicators,
    "baseload_indicators" = baseload_indicators,
    "daypart_values" = daypart_values)
  )
}


###
# Genetic Algorithm
###

decodeValueFromBin <- function(binary_representation, class_per_feature, nclasses_per_feature, 
                               levels_per_feature = NULL, min_per_feature = NULL, max_per_feature = NULL){
  
  bitOrders <- mapply(function(x) { nchar(toBin(x)) }, nclasses_per_feature)
  #binary_representation <- X
  binary_representation <- split(binary_representation, rep.int(seq.int(bitOrders), times = bitOrders))
  orders <- sapply(binary_representation, function(x) { binary2decimal(gray2binary(x)) })
  orders <- mapply(function(x){min(orders[x],nclasses_per_feature[x])},1:length(orders))
  orders <- mapply(
    function(x){
      switch(class_per_feature[x],
             "discrete"= levels_per_feature[[x]][orders[x]+1],
             "int"= floor(seq(min_per_feature[x],max_per_feature[x],
                              by=if(nclasses_per_feature[x]>0){
                                (max_per_feature[x]-min_per_feature[x])/(nclasses_per_feature[x])
                              }else{1}
             )[orders[x]+1]),
             "float"= seq(min_per_feature[x],max_per_feature[x],
                          by=if(nclasses_per_feature[x]>0){
                            (max_per_feature[x]-min_per_feature[x])/(nclasses_per_feature[x])
                          }else{1})[orders[x]+1]
      )
    }
    ,1:length(orders))
  return(unname(orders))
}

decodeBinFromValue <- function(values, class_per_feature, nclasses_per_feature,
                               levels_per_feature = NULL, min_per_feature = NULL, max_per_feature = NULL){
  # values=c(0,400)
  # class_per_feature=c("int","int")
  # nclasses_per_feature=c(4,5)
  # min_per_feature=c(0,0)
  # max_per_feature=c(400,500)
  #
  
  values <- mapply(
    function(x){
      
      switch(class_per_feature[x],
             "discrete"= which(levels_per_feature[[x]] %in% values[x])-1,
             "int"= which(seq(min_per_feature[x],max_per_feature[x],
                              by=(max_per_feature[x]-min_per_feature[x])/(nclasses_per_feature[x])) %in% values[x])-1,
             "float"= which(seq(min_per_feature[x],max_per_feature[x],
                                by=(max_per_feature[x]-min_per_feature[x])/(nclasses_per_feature[x])) %in% values[x])-1
      )
    }
    ,1:length(values))
  
  bitOrders <- mapply(function(x) { nchar(toBin(x)) }, nclasses_per_feature)
  binary_representation <- unlist(c(sapply(1:length(values), FUN=function(x) { binary2gray(decimal2binary(values[x],bitOrders[x])) })))
  
  return(binary_representation)
}

toBin<-function(x){ as.integer(paste(rev( as.integer(intToBits(x))),collapse="")) }

gaMonitor2 <- function (object, digits = getOption("digits"), ...)
{
  fitness <- na.exclude(object@fitness)
  cat(paste("GA | Iter =", object@iter, " | Mean =", format(mean(fitness),
                                                            digits = digits), " | Best =", format(max(fitness),
                                                                                                  digits = digits), "\n"))
  flush.console()
}


