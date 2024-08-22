#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

process SlideQC {

    input:
        val mpp
        val
        path slide

    output:
        path("*.qc.csv") into qc_ch

    script:
    """
    lazyslide qc $slide
    """
}