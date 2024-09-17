#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

params.slide_table = null
params.tile_px = 256
params.report_dir = "reports"
params.models = "resnet50"

process PREPROCESS {
    publishDir params.report_dir, mode: 'move'
    // conda "${projectDir}/env.yaml"

    input:
    tuple val(wsi), val(storage)
    val tile_px

    output:
    path '*_report.txt', emit: report
    tuple val(wsi), val(storage), emit: slide

    script:

    def wsi_base = wsi.baseName

    """
    lazyslide preprocess ${wsi} ${tile_px} --output ${storage}
    touch ${wsi_base}_report.txt
    """
}

process FEATURE {
    // conda "${projectDir}/env.yaml"

    input:
    tuple val(wsi), val(storage)
    each model

    script:
    """
    lazyslide feature ${wsi} ${model} --output ${storage}
    """
}



workflow {

    log.info """
    ██       █████  ███████ ██    ██ ███████ ██      ██ ██████  ███████
    ██      ██   ██    ███   ██  ██  ██      ██      ██ ██   ██ ██
    ██      ███████   ███     ████   ███████ ██      ██ ██   ██ █████
    ██      ██   ██  ███       ██         ██ ██      ██ ██   ██ ██
    ███████ ██   ██ ███████    ██    ███████ ███████ ██ ██████  ███████

    ===================================================================

    Workflow information:
    Workflow: ${workflow.projectDir}

    Input parameters:
    Slide table: ${file(params.slide_table)}

    """

    slides_ch = Channel
        .fromPath( params.slide_table, checkIfExists: true )
        .splitCsv( header: true )
        .map { row -> 
                def slide_file = file(row.file, checkIfExists: true)
                def slide_storage = row.storage
                if (row.storage == null) {  slide_storage = slide_file.parent / slide_file.baseName + ".zarr" }
                return tuple(slide_file, slide_storage)
         }

    // slides_ch.view()

    out_ch = PREPROCESS(slides_ch, params.tile_px)

    // println "Ouput of PREPROCESS: "
    // out_ch.slide.view()

    models = Channel.of(params.models?.split(','))

    FEATURE(out_ch.slide, models)

}