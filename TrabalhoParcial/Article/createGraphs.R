require(tidyverse)
require(mlbench)
source('./src/methods.R')

baseImgDir <- './Article/imgs/'
if(!dir.exists(baseImgDir)){
    dir.create(baseImgDir)
}

ggplot2::ggplot(data = NULL,
                ggplot2::aes(x = X[, 1], y = X[, 2],
                             color = log(Q), z = log(Q), alpha = log(Q) )) + 
    ggplot2::geom_point(size = 2.5) + 
    ggplot2::geom_density_2d(color = 'blue') +
    ggplot2::scale_colour_continuous(low = 'green', high = 'red') +
    ggplot2::theme_bw() +
    ggplot2::labs(x = 'X1', y = 'X2') +
    ggplot2::theme(axis.title.x = element_blank(),
                   axis.text.x = element_blank(),
                   axis.ticks.x = element_blank(),
                   axis.title.y = element_blank(),
                   axis.text.y = element_blank(),
                   axis.ticks.y = element_blank(),
                   legend.position = "none") +
    ggplot2::ggsave(file = paste(baseImgDir, 'qualidade_plot.png',sep = ''))

ggplot2::ggplot(data = NULL,
                ggplot2::aes(x = Q)) +
    geom_histogram(colour = "black",
                   fill = "white",
                   bins = 30) +
    ggplot2::theme_bw() +
    ggplot2::labs(x = 'Qualidade') +
    ggplot2::theme(axis.ticks.x = element_blank(),
                   axis.title.y = element_blank(),
                   axis.text.y = element_blank(),
                   axis.ticks.y = element_blank(),
                   legend.position = "none") +
    ggplot2::ggsave(file = paste(baseImgDir, 'histograma.png',sep = ''))

ggplot2::ggplot(data = NULL,
                ggplot2::aes(x = ' ', y = Q)) +
    ggplot2::geom_boxplot() +
    ggplot2::theme_bw() +
    ggplot2::labs(x = 'Qualidade') +
    ggplot2::theme(axis.ticks.x = element_blank(),
                   axis.title.y = element_blank(),
                   axis.text.y = element_blank(),
                   axis.ticks.y = element_blank(),
                   legend.position = "none") +
    ggplot2::ggsave(file = paste(baseImgDir, 'boxplot.png',sep = ''))



ggplot2::ggplot(data = NULL,
                ggplot2::aes(x = log(Q) )) +
    geom_histogram(colour = "black",
                   fill = "white",
                   bins = 30) +
    ggplot2::theme_bw() +
    ggplot2::labs(x = 'Log(Qualidade)') +
    ggplot2::theme(axis.ticks.x = element_blank(),
                   axis.title.y = element_blank(),
                   axis.text.y = element_blank(),
                   axis.ticks.y = element_blank(),
                   legend.position = "none") +
    ggplot2::ggsave(file = paste(baseImgDir, 'histograma_log.png',sep = ''))


ggplot2::ggplot(data = NULL,
                ggplot2::aes(x = ' ', y = log(Q))) +
    ggplot2::geom_boxplot() +
    ggplot2::theme_bw() +
    ggplot2::labs(x = 'Log(Qualidade)') +
    ggplot2::theme(axis.ticks.x = element_blank(),
                   axis.title.y = element_blank(),
                   axis.text.y = element_blank(),
                   axis.ticks.y = element_blank(),
                   legend.position = "none") +
    ggplot2::ggsave(file = paste(baseImgDir, 'boxplot_log.png',sep = ''))

