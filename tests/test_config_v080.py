from pathlib import Path
import yaml
from east_river_workflow.config import load_config


def test_config_keeps_single_noah_file_and_flexible_time(tmp_path):
    # Files do not need to exist for config loading.
    data={
      'project':{'analysis_start_datetime':'2018-01-01 00:00:00','analysis_end_datetime':'2018-02-01 23:00:00','water_years':[2018]},
      'paths':{
        'watershed_shapefile':'a.shp','isnobal_root_template':'x/{wy}','isnobal_topo_file':'topo.nc',
        'noahmp_output_file':'arbitrary_filename.nc','noahmp_geo_file':'geo.nc','snotel_csv':'s.csv','aso_directory':'aso','output_dir':str(tmp_path/'out')},
      'noahmp':{'expected_record_count':None}
    }
    p=tmp_path/'cfg.yaml'; p.write_text(yaml.safe_dump(data))
    cfg=load_config(p)
    assert cfg.data['paths']['noahmp_output_file']=='arbitrary_filename.nc'
    assert str(cfg.analysis_start)=='2018-01-01 00:00:00'
    assert cfg.section('noahmp')['daily_state_hour']==23


def test_config_accepts_numeric_noah_time_index(tmp_path):
    data={
      'project':{'analysis_start_datetime':'2017-10-01 00:00:00','analysis_end_datetime':'2018-06-30 23:00:00','water_years':[2018]},
      'paths':{
        'watershed_shapefile':'a.shp','isnobal_root_template':'x/{wy}','isnobal_topo_file':'topo.nc',
        'noahmp_output_file':'one_file.nc','noahmp_geo_file':'geo.nc','snotel_csv':'s.csv','aso_directory':'aso','output_dir':str(tmp_path/'out')},
      'noahmp':{
        'time_coordinate_mode':'numeric_index',
        'time_variable':'Time',
        'output_start_datetime':'2016-10-01 01:00:00',
        'expected_timestep_hours':1,
        'expected_record_count':85439,
      }
    }
    p=tmp_path/'numeric_cfg.yaml'; p.write_text(yaml.safe_dump(data))
    cfg=load_config(p)
    assert cfg.section('noahmp')['time_coordinate_mode']=='numeric_index'
    assert cfg.section('noahmp')['time_variable']=='Time'
    assert cfg.section('noahmp')['output_start_datetime']=='2016-10-01 01:00:00'
