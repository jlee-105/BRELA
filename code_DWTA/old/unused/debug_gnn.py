"""
GNN 디버깅 스크립트
"""

import torch
import traceback
from DWTA_GNN import create_gnn_actor

def debug_gnn():
    """GNN 각 단계별로 디버깅"""
    
    # 모델 로드
    model_path = "GNN_TRAIN_20250808_183546/CheckPoint_epoch00280/GNN_ACTOR_state_dic.pt"
    actor = create_gnn_actor()
    actor.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
    actor.eval()
    
    print("✅ 모델 로드 완료")
    
    # 30x30 입력 생성
    batch_size, para_size = 1, 1
    num_weapons, num_targets = 30, 30
    
    assignment = torch.randn(batch_size, para_size, num_weapons * num_targets + 1, 9)
    prob = torch.randn(batch_size, para_size, num_weapons, num_targets)
    mask = torch.ones(batch_size, para_size, num_weapons * num_targets + 1).bool()
    
    print(f"📊 입력 크기:")
    print(f"  assignment: {assignment.shape}")
    print(f"  prob: {prob.shape}")
    print(f"  mask: {mask.shape}")
    
    try:
        with torch.no_grad():
            # 단계별 디버깅
            print("\n🔍 Forward 시작...")
            
            # assignment_embedding에서 features 추출
            assignments = assignment[:, :, :-1, :]  # [1, 1, 900, 9]
            features_reshaped = assignments.view(batch_size, para_size, num_weapons, num_targets, 9)
            
            print(f"  assignments: {assignments.shape}")
            print(f"  features_reshaped: {features_reshaped.shape}")
            
            # Extract node and edge features
            weapon_features = features_reshaped[:, :, :, 0, :4]  # [1, 1, 30, 4]
            target_features = features_reshaped[:, :, 0, :, 4:8]  # [1, 1, 30, 4]
            edge_features = features_reshaped[:, :, :, :, 8:9]    # [1, 1, 30, 30, 1]
            
            print(f"  weapon_features: {weapon_features.shape}")
            print(f"  target_features: {target_features.shape}")
            print(f"  edge_features: {edge_features.shape}")
            
            # GNN layers 테스트
            weapon_h, target_h, edge_h = weapon_features, target_features, edge_features
            
            for i, layer in enumerate(actor.gnn_layers):
                print(f"\n  GNN Layer {i+1}:")
                print(f"    입력 - weapon_h: {weapon_h.shape}, target_h: {target_h.shape}, edge_h: {edge_h.shape}")
                
                weapon_h, target_h, edge_h = layer(weapon_h, target_h, edge_h)
                
                print(f"    출력 - weapon_h: {weapon_h.shape}, target_h: {target_h.shape}, edge_h: {edge_h.shape}")
            
            # Edge scorer 테스트
            print(f"\n  Edge scorer 입력: {edge_h.shape}")
            edge_scores = actor.edge_scorer(edge_h).squeeze(-1)
            print(f"  Edge scores: {edge_scores.shape}")
            
            edge_scores_flat = edge_scores.view(batch_size, para_size, num_weapons * num_targets)
            print(f"  Edge scores flat: {edge_scores_flat.shape}")
            
            # Global state 테스트
            global_weapon = weapon_h.mean(dim=2)  # [batch, para, hidden]
            global_target = target_h.mean(dim=2)   # [batch, para, hidden]
            print(f"  Global weapon: {global_weapon.shape}")
            print(f"  Global target: {global_target.shape}")
            
            global_state = torch.cat([global_weapon, global_target], dim=-1)
            print(f"  Global state: {global_state.shape}")
            
            no_action_score = actor.no_action_scorer(global_state).squeeze(-1)
            print(f"  No action score: {no_action_score.shape}")
            
            # Final output
            all_scores = torch.cat([edge_scores_flat, no_action_score.unsqueeze(-1)], dim=-1)
            print(f"  All scores: {all_scores.shape}")
            
            print("✅ 30x30 성공!")
            
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        print(f"에러 타입: {type(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_gnn() 