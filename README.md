log
- log_node1_1 : node1, OOOptimized_path_eps015.csv
  / 속도 12.0, a_lat=7로 테스트
  / 첫 유턴 구간에서 이탈
- log_node1_2 : node1, path_eps015_margin200
- log_node3_1 : node3, OOOptimized_path_eps015.csv (최종 1차 완주)
- log_node3_2 : node3, path_eps015_margin200
  / 진동이 많이 줄어보임, 176.74초

path_planning
  1. OOOptimized_path_eps015 : 최종 1차 완주한 경로
  2. path_eps015_margin200 : 1번 코드에서 유턴 부분 수정한 버전
  3. optimizeSplineRacingPathfinal.m : 경로 기본 함수
  4. gggenerate_racing_path_from_waypoints_final.m : 1번 경로 코드
  5. generate_racing_path_from.m_waypointsabc.m : 2번 경로 코드
 
src > mobile_racing_pkg > mobile_racing_pkg
  1. node 1 : 앞에 포인트 잡는거 수정한 버전
  2. node 3 : 최종 1차 완주
     * _2 붙은 코드는 로그 따는 부분 추가된 버전
