## GROUP 1 - CLUSTERING -------


###
## Clustering daily load curves ----
###

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
  normalization <- normalize_range_int(df_spread, 0, 1, norm_specs)
  df_spread <- normalization$norm
  
  # Generate the final spreated dataframe normalized
  complete_cases <- complete.cases(df_spread)
  if(filter_na==T){
    df_spread<- df_spread[complete_cases,]
  }
  
  return(list("raw_df"= df_agg,"norm_df"=df_spread, "norm_specs"=normalization$specs, "perc_cons"=perc_cons, "n_dayparts"=n_dayparts,
              "input_vars"= input_vars, "days_complete"=days[complete_cases]))
}

clustering_load_curves<-function(df, tz_local, time_column, value_column, temperature_column, perc_cons, n_dayparts, 
                                 norm_specs=NULL, input_vars, k=NULL, title="", output_plots=F, centroids_plot_file=NULL, bic_plot_file=NULL, 
                                 latex_font=F, plot_n_centroids_per_row=9, minimum_days_for_a_cluster=10, filename_prefix=NULL, force_plain_cluster=F){
  
  # df = df
  # group = "all"
  # tz_local = "Europe/Madrid"
  # time_column = "t"
  # value_column = "total_electricity"
  # temperature_column = "outdoor_temp"
  # k=2:10
  # perc_cons = T
  # n_dayparts = 24
  # norm_specs = NULL
  # input_vars = c("daily_cons","load_curves") # POSSIBLE INPUTS: c("load_curves", "days_weekend", "days_of_the_week", "daily_cons", "daily_temp"),
  # centroids_plot_file = "clustering.pdf"
  # bic_plot_file = "bic.pdf"
  # # centroids_plot_file = NULL,
  # # bic_plot_file = NULL,#"bic.pdf",
  # latex_font = F
  # plot_n_centroids_per_row=2
  # minimum_days_for_a_cluster = 10
  # force_plain_cluster = F
  # filename_prefix=paste(id,sep="~")
  
  # df_s <- split(df, df[,group])
  # 
  # results_all <- lapply(names(df_s),function(gr){
    
  #df <- df_s[[gr]]
  input_clust <- norm_load_curves(df, tz_local, time_column, value_column, temperature_column, perc_cons, n_dayparts, norm_specs, input_vars)
  
  # Initialize the objects
  # Test a clustering from 2 to 10 groups, if k is NULL (default).
  if(is.null(k)) k=seq(2,12)
  if((nrow(input_clust$norm_df)/4)<max(k)) k<-2:(nrow(input_clust$norm_df)/4)
  # Clustering model
  #mclustICL(apply(df_spread_norm,1:2,as.numeric),G = k, modelNames = c("EVI","VEI","EEI","VVI"))
  mclust_results <- Mclust(apply(input_clust$norm_df,1:2,as.numeric),G = k)#, modelNames=c("VEI","EEI","VVI","VII","EII"))
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
  
  # Deprecate those clusters with less than X days (X = minimum_days_for_a_cluster)
  important_clusters <- names(table(df_structural$s)[table(df_structural$s) >= 24*minimum_days_for_a_cluster])
  df_structural <- df_structural[df_structural$s %in% important_clusters,]
  df_structural$s <- as.character(factor(df_structural$s, levels=unique(df_structural$s), labels=as.character(1:length(unique(df_structural$s)))))
  
  # Reformat the cluster integer numbers to factors of 2 digits
  df_structural$s <- sprintf("%02i",as.integer(df_structural$s))
  
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
    centroids_plot_file <- paste(filename_prefix, centroids_plot_file,sep="~")  # Add gr when grouping...
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
                                axis.title=element_text(size=16))+labs(x="Hour of the day", y="Building hourly energy consumption [kWh]") +
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
                               axis.title=element_text(size=16))+labs(x="Hour of the day", y="Percentage of the total daily consumption") +
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
    bic_plot_file <- paste(filename_prefix, bic_plot_file,sep="~") # Add gr when grouping...
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
    
  # df_structural[,group] <- gr
  df_structural <- df_structural[order(df_structural[,"time"]),]
  
  # df_centroids_spread[,group] <- gr
  # df_centroids_avg[,group] <- gr
  
  return(list("df"=df_structural, "classified"=clustering_results, "centroids"=df_centroids_spread, "centroids_avg"=df_centroids_avg, "mod"=mclust_results, 
       "norm_specs"=input_clust$norm_specs, "perc_cons"=input_clust$perc_cons, "n_dayparts"=input_clust$n_dayparts, 
       "input_vars"= input_clust$input_vars))
  #})
  
  #names(results_all) <- names(df_s)
  
  #return(results_all)
}

classifier_load_curves <- function(df, df_centroids, clustering_mod, tz_local, time_column, value_column, temperature_column, 
                                   perc_cons, n_dayparts, filename_prefix,
                                   norm_specs=NULL, input_vars, plot_n_centroids_per_row=2, plot_file=NULL){
  
  # df_s <- split(df, df[,group])
  # df_centroids_ini <- df_centroids
  # 
  # results_all <- lapply(names(df_s),function(gr){
    
    # df_centroids <- df_centroids_ini[df_centroids_ini[,group]==gr,!(colnames(df_centroids_ini) %in% group)]
    
    # df <- df_s[[gr]]
    
    input_class <- norm_load_curves(df, tz_local, time_column, value_column, temperature_column, perc_cons, n_dayparts, norm_specs, input_vars,filter_na = F)
    
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
    
    #shapes_1 <- predict(clustering_mod[[gr]],input_class$norm_df)$classification
    shapes_1 <- dist_shapes %>% apply(1, function(x) {ifelse(sum(is.na(x))>1, NA, order(x)[1])})
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
      plot_file <- paste(filename_prefix, "classification.pdf",sep="~")  # Add gr when grouping...
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
        labs(x="Hour of the day", y="Percentage of the total daily consumption") +
        theme(text= element_text(size=16))+
        scale_x_continuous(
          breaks = c(0, 12, 23),
          label = c("00:00","12:00","23:00")
        )
      ggsave(plot_file,p,height = 6.5,width = 6.5)
    }
    # df_consumption_test[,group] <- gr
    # df_consumption_test
    
  # })
  
  # names(results_all) <- names(df_s)
  
  # return(results_all)
  return(df_consumption_test)
}


# GROUP 2 CHARACTERISATION ----
add_cooling_heating_daily_deltaT_features <- function(df, temperature_column="temperature", daypart_column="daypart",# seasonality_column="s",
                                                      tbal=19, hysteresis=1, hours_of_each_daypart=8){
  
  tbal_vector <- tbal[as.integer(as.character(df[,daypart_column]))+1]
  #tbal[(as.integer(as.character(df[,daypart_column]))+1)+max(as.integer(as.character(df[,daypart_column]))+1)*(as.numeric(as.character(df$s))-1)]
  hysteresis_vector <- hysteresis[as.integer(as.character(df[,daypart_column]))+1]
  #hysteresis[(as.integer(as.character(df[,daypart_column]))+1)+max(as.integer(as.character(df[,daypart_column]))+1)*(as.numeric(as.character(df$s))-1)]
  
  df[,paste0(temperature_column,"_h")] <- ifelse(
    (tbal_vector-hysteresis_vector) >= df[,temperature_column],
    (tbal_vector-hysteresis_vector) - df[,temperature_column], 0)
  df[,paste0(temperature_column,"_status_h")] <- ifelse(
    (tbal_vector-hysteresis_vector) >= df[,temperature_column],
    1, 0)
  if(!is.null(daypart_column)){
    for(s_ in unique(as.character(df[,daypart_column]))){
      if(sum(df[df[,daypart_column]==s_,paste0(temperature_column,"_h")],na.rm=T) <= 50){
        df[df[,daypart_column]==s_, paste0(temperature_column,"_h")] <- 0
        df[df[,daypart_column]==s_, paste0(temperature_column,"_status_h")] <- 0
      }
    }
  } else {
    if(sum(df[,paste0(temperature_column,"_h")],na.rm=T) <= 100){
      df[, paste0(temperature_column,"_h")] <- 0
      df[, paste0(temperature_column,"_status_h")] <- 0
    }
  }
  df[,paste0(temperature_column,"_c")] <- ifelse(
    (tbal_vector+hysteresis_vector) <= df[,temperature_column],
    df[,temperature_column] - (tbal_vector+hysteresis_vector), 0)
  df[,paste0(temperature_column,"_status_c")] <- ifelse(
    (tbal_vector+hysteresis_vector) <= df[,temperature_column],
    1, 0)
  if(!is.null(daypart_column)){
    for(s_ in unique(as.character(df[,daypart_column]))){
      if(sum(df[df[,daypart_column]==s_,paste0(temperature_column,"_c")],na.rm=T) <= 50){
        df[df[,daypart_column]==s_, paste0(temperature_column,"_c")] <- 0
        df[df[,daypart_column]==s_, paste0(temperature_column,"_status_c")] <- 0
      }
    }
  } else {
    if(sum(df[,paste0(temperature_column,"_c")],na.rm=T) <= 100){
      df[, paste0(temperature_column,"_c")] <- 0
      df[, paste0(temperature_column,"_status_c")] <- 0
    }
  }
  
  return(df)
}


add_cooling_heating_daily_GHI_features <- function(df, temperature_column="temperature", GHI_column="GHI"){
  
  df[,GHI_column] <- ifelse(df[,GHI_column]<50, 0, df[,GHI_column])
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

add_fs_daypart <- function(df, time_column, hours_of_each_daypart){
  df$hour_daypart <- hour(df[,time_column])#((as.numeric(as.character(df[,daypart_column]))*hours_of_each_daypart)+hours_of_each_daypart/2)
  df_fs <- do.call(cbind,
                   fs(ifelse(df$hour_daypart>=12,df$hour_daypart/24-0.5,df$hour_daypart/24+0.5),
                      nharmonics = 5, odd = F, prefix="daypart_fs_"))
  df <- cbind(df,df_fs) 
}

yhour <- function(time) {
  (yday(time) - 1) * 24 + hour(time)
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

characterization_model_trainer <- function(params, df, temperature_column = "temperature", value_column = "value", windSpeed_column = "windSpeed", 
                                           daypart_column = "daypart", time_column= "time", GHI_column = "GHI", intercept_column = "weekday", for_optimize=F, 
                                           hours_of_each_daypart=8,rows_to_train=NULL){
  error <- F
  tryCatch({
    #params <- c(characterization$mod@nuisance$tbal, characterization$mod@nuisance$hysteresis)
    if(!is.null(rows_to_train)){
      df$val <- !(rownames(df) %in% as.character(rows_to_train))
    } else {
      df$val <- F
    }
    df <- tune_with_params(df = df, params = params, temperature_column = temperature_column, GHI_column = GHI_column, 
                           windSpeed_column = windSpeed_column, daypart_column = daypart_column, weekday_column = intercept_column, 
                           time_column = time_column, seasonality_column = "s", hours_of_each_daypart = hours_of_each_daypart,
                           training_seasonality_levels = NULL)
    
    if(length(unique(df[,daypart_column]))>1){
      form <- as.formula(sprintf("%s ~ 0 + %s" #+ 
                                 #%s_h:%s + %s_c:%s"# + %s_h:%s + %s_c:%s" 
                                 ,value_column
                                 ,paste(
                                   "s"
                                   , paste0(colnames(df)[grepl("^daypart_fs_",colnames(df))],":s",collapse=" + ")
                                   , sprintf("%s_status_c:daypart",temperature_column)
                                   , sprintf("%s_c:daypart",temperature_column)
                                   , sprintf("%s_status_h:daypart",temperature_column)
                                   , sprintf("%s_h:daypart",temperature_column)
                                   ,sep=" + ")
      ))
    }
    
    # LS version
    # mod2 <- lm(formula = form, data = df)
    y <- model.frame(form,df[df$val==F,])[,1]
    x <- as.matrix(model.matrix(form,df[df$val==F,]))
    mod_pen <- tryCatch({
      penalized(
        y,x,~0,positive = grepl("outdoor_temp|GHI|windSpeed",colnames(x)),
        lambda1 = 0,lambda2 = 0,
        #startbeta = ifelse(grepl("temperature|GHI|windSpeed",colnames(x)),0,1),
        trace=F
      )
    }, error=function(e){
      penalized(
        y,x,~0,positive = grepl("temperature|GHI|windSpeed",colnames(x)),
        lambda1 = 0.2,lambda2 = 0.2,
        #startbeta = ifelse(grepl("temperature|GHI|windSpeed",colnames(x)),0,1),
        trace=F
      )
    })
    mod_pen@nuisance$tbal <- params[1:(length(params)/2)]
    mod_pen@nuisance$hysteresis <- params[((length(params)/2)+1):length(params)]
    mod_pen@nuisance$model <- df[df$val==F,]
    mod_pen@nuisance$formula <- form
    mod_pen@nuisance$seasonality_levels <- levels(df[df$val==F,"s"])
    
    df <- df[complete.cases(df),]
    df$pred <- unlist(as.matrix(model.matrix(mod_pen@nuisance$formula[-2],df))[,names(coef(mod_pen))]) %*% coef(mod_pen)
    df$pred <- ifelse(df$pred > 0, df$pred, 0)
    
    # plotly::ggplotly(ggplot(df)+geom_line(aes(1:nrow(df),total_electricity))+geom_line(aes(1:nrow(df),pred),col="red",alpha=0.5))

  }, error=function(e){error<<-T})
  
  if(for_optimize==T){
    
    if(error == T){
      return(-10000)
    } else {
      
      # Filter the days without weather dependency
      dfv <- df[
        is.finite(df$pred) & is.finite(df[,value_column]),# & df$val==T,
        ]
      
      return(
        -rmserr(dfv$pred,dfv[,value_column])$rmse
      )
      
    }
    
  } else {
    return(list("df"=df, "mod"=mod_pen))
  }
  
}

tune_with_params <- function(df, params, temperature_column, GHI_column, windSpeed_column, daypart_column, weekday_column, time_column, 
                             seasonality_column = "s", force_weather_heating_vars_to_0 = F, force_weather_cooling_vars_to_0 = F, 
                             hours_of_each_daypart=8, training_seasonality_levels=NULL){
  
  if(!is.null(training_seasonality_levels)){
    df[,seasonality_column] <- factor(df[,seasonality_column], levels=training_seasonality_levels)
  }
  
  tbal <- params[1:(length(params)/2)]
  hysteresis <- params[((length(params)/2)+1):length(params)]
  
  df <- add_cooling_heating_daily_deltaT_features(
    df = df,temperature_column = temperature_column, tbal = tbal, hysteresis = hysteresis, 
    daypart_column = daypart_column, hours_of_each_daypart = hours_of_each_daypart
  )
  df <- add_cooling_heating_daily_GHI_features(
    df = df,temperature_column = temperature_column, GHI_column = GHI_column
  )
  df <- add_cooling_heating_daily_windSpeed_features(
    df = df,temperature_column = temperature_column, windSpeed_column = windSpeed_column
  )
  df <- add_fs_daypart(
    df = df, time_column = time_column,hours_of_each_daypart = hours_of_each_daypart
  )
  if (force_weather_heating_vars_to_0==T) {
    df[,grepl(paste(paste0(temperature_column,'_h'), paste0(GHI_column,'_h'), paste0(windSpeed_column,'_h'), sep="|"), colnames(df))] <- 0
  }
  if (force_weather_cooling_vars_to_0==T) {
    df[,grepl(paste(paste0(temperature_column,'_c'), paste0(GHI_column,'_c'), paste0(windSpeed_column,'_c'), sep="|"), colnames(df))] <- 0
  }
  return(df)
}

characterizer <- function(df_ini, value_column="value", temperature_column = "temperature", windSpeed_column = "windSpeed", 
                          GHI_column = "GHI", time_column = "time", intercept_column="weekday", tz_local="Europe/Madrid", 
                          group_column="season", hours_of_each_daypart=8, centroids = df_centroids, centroids_summary = df_centroids_avg,
                          classification = classification[,c("date","s","tariff")]){
  
  # df_ini = df
  # tz_local = "Europe/Madrid"
  # time_column = "t"
  # value_column = "total_electricity"
  # temperature_column = "outdoor_temp"
  # GHI_column = "GHI"
  # intercept_column = "weekday"
  # windSpeed_column = "windSpeed"
  # group_column ="all"
  # hours_of_each_daypart = 4
  # centroids = df_centroids
  # centroids_summary = df_centroids_avg
  # classification = classification[,c("date","s")]
  
  df <- df_ini
  
  df$daypart <- as.character(floor(hour(with_tz(df[,time_column],tz=tz_local))/hours_of_each_daypart))
  
  tbal_min <- rep(quantile(df[,temperature_column],0.25),(24/hours_of_each_daypart)) #*length(unique(daily_df$s))
  tbal_max <- rep(quantile(df[,temperature_column],0.75),(24/hours_of_each_daypart))
  hysteresis_min <- rep(0,(24/hours_of_each_daypart))
  hysteresis_max <- rep(7,(24/hours_of_each_daypart))
  
  if(identical(df,list())){
    return(NULL)
  }
  
  rows_to_train <- do.call(c,lapply(unique(df$s),
                                    FUN= function(x){
                                      sample(as.numeric(rownames(df[df$s==x,])),
                                             nrow(df[df$s==x,])*0.8,replace = F)
                                    }))
    
  # Least squares
  GA <- ga(type = "real-valued",
           fitness = characterization_model_trainer,
           lower = c(tbal_min,hysteresis_min), upper = c(tbal_max,hysteresis_max),
           df = df,
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
           monitor=gaMonitor2,
           suggestions = t(data.frame(c(tbal_min,hysteresis_min),c(tbal_max,hysteresis_max))),
           maxiter=10,popSize = 32,parallel= F,pmutation = 0.1) #monitor = monitor)
  tbal <- GA@solution[1,(1:(ncol(GA@solution)/2))]
  hysteresis <- GA@solution[1,((ncol(GA@solution)/2+1):ncol(GA@solution))]
  
  tbal_min <- tbal - 1
  tbal_max <- tbal + 1
  hysteresis_min <- hysteresis-0.5
  hysteresis_max <- hysteresis+0.5
  hysteresis_min[hysteresis_min<0] <- 0
  hysteresis_max[hysteresis_max>7] <- 7
  
  daily_model <- characterization_model_trainer(
    params = c(tbal,hysteresis), 
    temperature_column = temperature_column,
    windSpeed_column = windSpeed_column,
    GHI_column = GHI_column,
    value_column = value_column,
    intercept_column = intercept_column,
    time_column = time_column,
    for_optimize = F,
    daypart_column = "daypart",
    df = df,
    hours_of_each_daypart = hours_of_each_daypart,
    rows_to_train = NULL)

  # Prediction dataframe output
  results_all <- list(
    "df" = daily_model$df,
    "mod" = daily_model$mod
  )

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


