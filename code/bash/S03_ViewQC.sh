#!/bin/tcsh
ml firefox
# Construct a list of all subjects to view
set all_files = `\ls /data/SFIMJGC_HCP7T/BCBL2024/prcs_data/sub-[12]?/D02_Preproc_fMRI_ses-1/QC_sub-[12]?/index.html`
echo $#all_files
# Loop over all the subjects in the list
foreach ii ( `seq 1 1 $#all_files` )

    set ff = ${all_files[$ii]}
    echo "++ Opening: $ff"
    sleep 0.1      # this helps *all* windows open properly

    # Open the first HTML a new window, the rest in a new tab
    if ( $ii == 1 ) then
        firefox -new-window ${ff}\#vstat
    else
        firefox -new-tab    ${ff}\#vstat
    endif

end
