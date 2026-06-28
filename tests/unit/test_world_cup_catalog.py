from worldcup.data_ingestion.sources.world_cup_catalog import WORLD_CUP_2018, WORLD_CUP_2022


def test_world_cup_catalog_has_full_tournaments():
    assert len(WORLD_CUP_2018) == 64
    assert len(WORLD_CUP_2022) == 64
