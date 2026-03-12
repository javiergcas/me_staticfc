Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ01_constant_gated_task-rest -gert_chan_prefix _echo -gert_write_as_nifti -gert_quit_on_err  -num_chan 3 -sort_method geme_index -infile_pattern "mr_0013/*dcm"
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ01_cardiac_gated_task-rest  -gert_chan_prefix _echo -gert_write_as_nifti -gert_quit_on_err  -num_chan 3 -sort_method geme_index -infile_pattern "mr_0011/*dcm"
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ01_Anat -gert_write_as_nifti -gert_quit_on_err -sort_method geme_index -infile_pattern "" 

# MGSBJ01
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ01_ses-1_T1w.nii.gz         -gert_write_as_nifti -gert_quit_on_err -infile_pattern "3-Sagittal_Anat_MP_Rage_1_mm/resources/DICOM/files/sagittal_anat_mp_rage_1_mm-00*"
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ01_cardiac_gated_task-rest  -gert_chan_prefix _echo -gert_write_as_nifti -gert_quit_on_err  -num_chan 3 -sort_method geme_index -infile_pattern "11-EPI3isoMEGY___Rest/resources/DICOM/files/*dcm"
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ01_constant_gated_task-rest -gert_chan_prefix _echo -gert_write_as_nifti -gert_quit_on_err  -num_chan 3 -sort_method geme_index -infile_pattern "13-EPI3isoMEGN___Rest/resources/DICOM/files/*dcm"

# MGSBJ02
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ02_ses-1_T1w.nii.gz         -gert_write_as_nifti -gert_quit_on_err -infile_pattern "8-Sagittal_Anat_MP_Rage_1_mm/resources/DICOM/files/*dcm"
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ02_cardiac_gated_task-rest -gert_chan_prefix _echo -gert_write_as_nifti -gert_quit_on_err  -num_chan 3 -sort_method geme_index -infile_pattern "10-EPI3isoMEGY___Rest/resources/DICOM/files/epi3isomegy_rest-0*dcm"
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ02_constant_gated_task-rest -gert_chan_prefix _echo -gert_write_as_nifti -gert_quit_on_err  -num_chan 3 -sort_method geme_index -infile_pattern "11-EPI3isoMEGN___Rest/resources/DICOM/files/*dcm"
cp MR5000385727_20160429_101736/resources/supplement/realtime/Trig_log_epiRTnihVR01_scan_0012_20160429_111151.1D MGSBJ02_cardiac_gated_task-rest.Triggers.1D
cp MR5000385727_20160429_101736/resources/supplement/realtime/Trig_log_epiRTnihVR01_scan_0013_20160429_112434.1D MGSBJ02_constant_gated_task-rest.Triggers.1D

#MGSBJ03
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ03_ses-1_T1w.nii.gz         -gert_write_as_nifti -gert_quit_on_err -infile_pattern "8-Sagittal_Anat_MP_Rage_1_mm/resources/DICOM/files/*dcm"
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ03_cardiac_gated_task-rest -gert_chan_prefix _echo -gert_write_as_nifti -gert_quit_on_err  -num_chan 3 -sort_method geme_index -infile_pattern "11-EPI3isoMEGY___Rest/resources/DICOM/files/*dcm"
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ03_constant_gated_task-rest -gert_chan_prefix _echo -gert_write_as_nifti -gert_quit_on_err  -num_chan 3 -sort_method geme_index -infile_pattern "10-EPI3isoMEGN___Rest/resources/DICOM/files/*dcm"
cp MR5000387048_20160503_203026/resources/supplement/realtime/Trig_log_epiRTnihVR01_scan_0012_20160503_212418.1D MGSBJ03_constant_gated_task-rest.Triggers.1D
cp MR5000387048_20160503_203026/resources/supplement/realtime/Trig_log_epiRTnihVR01_scan_0013_20160503_213531.1D MGSBJ03_cardiac_gated_task-rest.Triggers.1D

#MGSBJ04
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ04_ses-1_T1w.nii.gz         -gert_write_as_nifti -gert_quit_on_err -infile_pattern "9-Sagittal_Anat_MP_Rage_1_mm/resources/DICOM/files/*dcm"
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ04_cardiac_gated_task-rest -gert_chan_prefix _echo -gert_write_as_nifti -gert_quit_on_err  -num_chan 3 -sort_method geme_index -infile_pattern "12-EPI3isoMEGY___Rest/resources/DICOM/files/*dcm"
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ04_constant_gated_task-rest -gert_chan_prefix _echo -gert_write_as_nifti -gert_quit_on_err  -num_chan 3 -sort_method geme_index -infile_pattern "13-EPI3isoMEGN___Rest/resources/DICOM/files/*dcm"

#MGSBJ05
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ05_ses-1_T1w.nii.gz         -gert_write_as_nifti -gert_quit_on_err -infile_pattern "8-Sagittal_Anat_MP_Rage_1_mm/resources/DICOM/files/*dcm"
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ05_cardiac_gated_task-rest -gert_chan_prefix _echo -gert_write_as_nifti -gert_quit_on_err  -num_chan 3 -sort_method geme_index -infile_pattern "11-EPI3isoMEGY___Rest/resources/DICOM/files/*dcm"
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ05_constant_gated_task-rest -gert_chan_prefix _echo -gert_write_as_nifti -gert_quit_on_err  -num_chan 3 -sort_method geme_index -infile_pattern "10-EPI3isoMEGN___Rest/resources/DICOM/files/*dcm"
cp MR5000387138_20160506_082421/resources/supplement/realtime/Trig_log_epiRTnihVR01_scan_0014_20160506_092939.1D MGSBJ05_cardiac_gated_task-rest.Triggers.1D
cp MR5000387138_20160506_082421/resources/supplement/realtime/Trig_log_epiRTnihVR01_scan_0013_20160506_091919.1D MGSBJ05_constant_gated_task-rest.Triggers.1D

#MGSBJ06
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ06_ses-1_T1w.nii.gz         -gert_write_as_nifti -gert_quit_on_err -infile_pattern "8-Sagittal_Anat_MP_Rage_1_mm/resources/DICOM/files/*dcm"
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ06_cardiac_gated_task-rest -gert_chan_prefix _echo -gert_write_as_nifti -gert_quit_on_err  -num_chan 3 -sort_method geme_index -infile_pattern "10-EPI3isoMEGY___Rest/resources/DICOM/files/*dcm"
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ06_constant_gated_task-rest -gert_chan_prefix _echo -gert_write_as_nifti -gert_quit_on_err  -num_chan 3 -sort_method geme_index -infile_pattern "11-EPI3isoMEGN___Rest/resources/DICOM/files/*dcm"
cp MR5000387135_20160506_102153/resources/supplement/realtime/Trig_log_epiRTnihVR01_scan_0012_20160506_111143.1D MGSBJ06_cardiac_gated_task-rest.Triggers.1D
cp MR5000387135_20160506_102153/resources/supplement/realtime/Trig_log_epiRTnihVR01_scan_0013_20160506_112409.1D MGSBJ06_constant_gated_task-rest.Triggers.1D

#MGSBJ07
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ07_ses-1_T1w.nii.gz         -gert_write_as_nifti -gert_quit_on_err -infile_pattern "8-Sagittal_Anat_MP_Rage_1_mm/resources/DICOM/files/*dcm"
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ07_cardiac_gated_task-rest -gert_chan_prefix _echo -gert_write_as_nifti -gert_quit_on_err  -num_chan 3 -sort_method geme_index -infile_pattern "11-EPI3isoMEGY___Rest/resources/DICOM/files/*dcm"
Dimon -dicom_org -gert_create_dataset -gert_to3d_prefix MGSBJ07_constant_gated_task-rest -gert_chan_prefix _echo -gert_write_as_nifti -gert_quit_on_err  -num_chan 3 -sort_method geme_index -infile_pattern "10-EPI3isoMEGN___Rest/resources/DICOM/files/*dcm"
