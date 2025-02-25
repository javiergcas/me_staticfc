Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ01_constant_gated_task-rest -gert_chan_prefix _echo -gert_write_as_nifti -gert_quit_on_err  -num_chan 3 -sort_method geme_index -infile_pattern "mr_0013/*dcm"
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ01_cardiac_gated_task-rest  -gert_chan_prefix _echo -gert_write_as_nifti -gert_quit_on_err  -num_chan 3 -sort_method geme_index -infile_pattern "mr_0011/*dcm"
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ01_Anat -gert_write_as_nifti -gert_quit_on_err -sort_method geme_index -infile_pattern ""  
