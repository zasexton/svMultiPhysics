#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include "Topology/CellTopology.h"

using namespace svmp;

static long long pack_edge(index_t a, index_t b) {
  index_t lo = std::min(a,b); index_t hi = std::max(a,b);
  return (static_cast<long long>(lo) << 32) | static_cast<unsigned long long>(static_cast<uint32_t>(hi));
}

static bool edge_orientations_cancel(const std::vector<std::vector<index_t>>& faces, std::string& msg) {
  std::unordered_map<long long,int> sum;
  for (const auto& f : faces) {
    size_t k=f.size();
    for (size_t i=0;i<k;++i){ index_t u=f[i], v=f[(i+1)%k]; sum[pack_edge(u,v)] += (u<v)?+1:-1; }
  }
  for (auto& kv : sum) { if (kv.second != 0) { msg = "edge sum nonzero"; return false; } }
  return true;
}

static bool check_face_edges(CellFamily fam, std::string& msg){
  auto fview = CellTopology::get_oriented_boundary_faces_view(fam);
  if (!(fview.indices && fview.offsets && fview.face_count>0)) fview = CellTopology::get_boundary_faces_canonical_view(fam);
  auto eview = CellTopology::get_edges_view(fam);
  if (!(fview.indices && fview.offsets && fview.face_count>0 && eview.pairs_flat && eview.edge_count>0)) {
    if (fam==CellFamily::Triangle || fam==CellFamily::Quad) return true; // 2D fallback
    msg = "views missing"; return false;
  }
  std::unordered_map<long long,int> emap;
  for (int ei=0;ei<eview.edge_count;++ei) emap.emplace(pack_edge(eview.pairs_flat[2*ei], eview.pairs_flat[2*ei+1]), ei);
  auto mapped = CellTopology::get_face_edges(fam);
  if ((int)mapped.size() != fview.face_count) { msg = "size mismatch"; return false; }
  for (int fi=0; fi<fview.face_count; ++fi){
    int b=fview.offsets[fi], e=fview.offsets[fi+1];
    int fv=e-b;
    std::vector<int> expected;
    if (fv==2){ auto it=emap.find(pack_edge(fview.indices[b], fview.indices[b+1])); if (it==emap.end()) { msg="2D edge not found"; return false; } expected={it->second}; }
    else{
      for(int k=0;k<fv;++k){ index_t u=fview.indices[b+k], v=fview.indices[b+((k+1)%fv)]; auto it=emap.find(pack_edge(u,v)); if (it==emap.end()){ msg="3D edge not found"; return false; } expected.push_back(it->second);} }
    if (expected != mapped[fi]) { msg = "edge list mismatch"; return false; }
  }
  return true;
}

int main(){
  int fails = 0; auto fail=[&](const char* name, const std::string& m){ std::cout<<"[FAIL] "<<name<<": "<<m<<"\n"; ++fails; };

  // Face->edge mapping for fixed families
  for (auto fam : {CellFamily::Triangle, CellFamily::Quad, CellFamily::Tetra, CellFamily::Hex, CellFamily::Wedge, CellFamily::Pyramid}){
    std::string msg; if (!check_face_edges(fam,msg)) fail("face_edges", msg);
  }

  // Oriented edge cancellation for 3D
  for (auto fam : {CellFamily::Tetra, CellFamily::Hex, CellFamily::Wedge, CellFamily::Pyramid}){
    auto faces = CellTopology::get_oriented_boundary_faces(fam);
    std::string msg; if (!edge_orientations_cancel(faces,msg)) fail("edge_cancel", msg);
  }

  // Prism(m=3) matches Wedge
  {
    int m=3; auto pfv = CellTopology::get_prism_faces_view(m);
    std::vector<std::vector<index_t>> prism_faces; for(int f=0; f<pfv.face_count; ++f){ int b=pfv.offsets[f], e=pfv.offsets[f+1]; prism_faces.emplace_back(pfv.indices+b, pfv.indices+e);} 
    auto wedge_faces = CellTopology::get_oriented_boundary_faces(CellFamily::Wedge);
    if (prism_faces != wedge_faces) fail("prism_m3_match", "faces differ");
  }

  // Pyramid pattern counts and inference
  for (int p : {3,4,5}){
    auto pat = CellTopology::high_order_pattern(CellFamily::Pyramid, p, CellTopology::HighOrderKind::Lagrange);
    size_t corners=5, edges = 8*(p-1), faces_tri = 4*((p-1)*(p-2)/2), face_quad=(p-1)*(p-1); size_t vol = (p>=3)? (size_t)((p-2)*(p-1)*(2*p-3)/6):0; size_t expected = corners+edges+faces_tri+face_quad+vol;
    if (pat.sequence.size() != expected) fail("pyr_pattern_count", std::string("p=")+std::to_string(p)+", got="+std::to_string(pat.sequence.size())+", exp="+std::to_string(expected));
  }

  // Lagrange order inference
  if (CellTopology::infer_lagrange_order(CellFamily::Wedge,6)!=1 || CellTopology::infer_lagrange_order(CellFamily::Wedge,18)!=2 || CellTopology::infer_lagrange_order(CellFamily::Wedge,40)!=3) fail("wedge_infer","bad n->p");
  if (CellTopology::infer_lagrange_order(CellFamily::Pyramid,13)!=2 || CellTopology::infer_lagrange_order(CellFamily::Pyramid,14)!=2 || CellTopology::infer_lagrange_order(CellFamily::Pyramid,30)!=3 || CellTopology::infer_lagrange_order(CellFamily::Pyramid,55)!=4) fail("pyr_infer","bad n->p");

  // Serendipity quad inference
  if (CellTopology::infer_serendipity_order(CellFamily::Quad,8)!=2 || CellTopology::infer_serendipity_order(CellFamily::Quad,12)!=3 || CellTopology::infer_serendipity_order(CellFamily::Quad,16)!=4 || CellTopology::infer_serendipity_order(CellFamily::Quad,20)!=5 || CellTopology::infer_serendipity_order(CellFamily::Quad,10)!=-1) fail("quad_ser_infer","bad n->p");

  // Face list order checks: Hex/Wedge/Pyramid (first faces)
  {
    auto hex = CellTopology::get_oriented_boundary_faces_view(CellFamily::Hex);
    if (hex.face_count!=6) fail("hex_facecount","!=6");
    // bottom then top
    if (!(hex.indices[hex.offsets[0]+0]==0 && hex.indices[hex.offsets[0]+1]==3 && hex.indices[hex.offsets[0]+2]==2 && hex.indices[hex.offsets[0]+3]==1)) fail("hex_face0","mismatch");
    if (!(hex.indices[hex.offsets[1]+0]==4 && hex.indices[hex.offsets[1]+1]==5 && hex.indices[hex.offsets[1]+2]==6 && hex.indices[hex.offsets[1]+3]==7)) fail("hex_face1","mismatch");
    auto wed = CellTopology::get_oriented_boundary_faces_view(CellFamily::Wedge);
    if (wed.face_count!=5) fail("wedge_facecount","!=5");
    if (!(wed.indices[wed.offsets[0]+0]==0 && wed.indices[wed.offsets[0]+1]==2 && wed.indices[wed.offsets[0]+2]==1)) fail("wedge_face0","mismatch");
    if (!(wed.indices[wed.offsets[1]+0]==3 && wed.indices[wed.offsets[1]+1]==4 && wed.indices[wed.offsets[1]+2]==5)) fail("wedge_face1","mismatch");
    auto pyr = CellTopology::get_oriented_boundary_faces_view(CellFamily::Pyramid);
    if (pyr.face_count!=5) fail("pyr_facecount","!=5");
    if (!(pyr.indices[pyr.offsets[0]+0]==0 && pyr.indices[pyr.offsets[0]+1]==1 && pyr.indices[pyr.offsets[0]+2]==2 && pyr.indices[pyr.offsets[0]+3]==3)) fail("pyr_face0","mismatch");
  }

  // 2D CCW orientation explicit: Triangle and Quad
  {
    auto tri = CellTopology::get_oriented_boundary_faces_view(CellFamily::Triangle);
    if (!(tri.indices[tri.offsets[0]+0]==0 && tri.indices[tri.offsets[0]+1]==1)) fail("tri_edge0","mismatch");
    if (!(tri.indices[tri.offsets[1]+0]==1 && tri.indices[tri.offsets[1]+1]==2)) fail("tri_edge1","mismatch");
    if (!(tri.indices[tri.offsets[2]+0]==2 && tri.indices[tri.offsets[2]+1]==0)) fail("tri_edge2","mismatch");
    auto quad = CellTopology::get_oriented_boundary_faces_view(CellFamily::Quad);
    if (!(quad.indices[quad.offsets[0]+0]==0 && quad.indices[quad.offsets[0]+1]==1)) fail("quad_edge0","mismatch");
    if (!(quad.indices[quad.offsets[1]+0]==1 && quad.indices[quad.offsets[1]+1]==2)) fail("quad_edge1","mismatch");
    if (!(quad.indices[quad.offsets[2]+0]==2 && quad.indices[quad.offsets[2]+1]==3)) fail("quad_edge2","mismatch");
    if (!(quad.indices[quad.offsets[3]+0]==3 && quad.indices[quad.offsets[3]+1]==0)) fail("quad_edge3","mismatch");
  }

  // High‑order per‑face interior ordering checks
  auto face_roles = [](const CellTopology::HighOrderPattern& pat, int fid){ std::vector<std::pair<int,int>> out; for (auto& r: pat.sequence) if (r.role==CellTopology::HONodeRole::Face && r.idx0==fid) out.emplace_back(r.idx1,r.idx2); return out; };
  // Tetra p=4: tri faces: (1,1),(1,2),(2,1)
  {
    int p=4; auto pat = CellTopology::high_order_pattern(CellFamily::Tetra,p, CellTopology::HighOrderKind::Lagrange); auto fv = CellTopology::get_oriented_boundary_faces_view(CellFamily::Tetra);
    std::vector<std::pair<int,int>> exp={{1,1},{1,2},{2,1}};
    for (int fi=0; fi<fv.face_count; ++fi) if (fv.offsets[fi+1]-fv.offsets[fi]==3) { if (face_roles(pat,fi)!=exp) fail("tet_face_order","tri face lexicographic"); }
  }
  // Hex p=3: quad faces row‑major i=1..2, j=1..2
  {
    int p=3; auto pat=CellTopology::high_order_pattern(CellFamily::Hex,p, CellTopology::HighOrderKind::Lagrange); auto fv=CellTopology::get_oriented_boundary_faces_view(CellFamily::Hex);
    std::vector<std::pair<int,int>> exp; for (int i=1;i<=p-1;++i) for (int j=1;j<=p-1;++j) exp.emplace_back(i,j);
    for (int fi=0; fi<fv.face_count; ++fi) if (fv.offsets[fi+1]-fv.offsets[fi]==4) { if (face_roles(pat,fi)!=exp) fail("hex_face_order","quad face row‑major"); }
  }
  // Wedge p=3: mix of tri/quad faces
  {
    int p=3; auto pat=CellTopology::high_order_pattern(CellFamily::Wedge,p, CellTopology::HighOrderKind::Lagrange); auto fv=CellTopology::get_oriented_boundary_faces_view(CellFamily::Wedge);
    for (int fi=0; fi<fv.face_count; ++fi) {
      int sz=fv.offsets[fi+1]-fv.offsets[fi]; auto got=face_roles(pat,fi);
      if (sz==3){ std::vector<std::pair<int,int>> exp; for (int i=1;i<=p-2;++i) for (int j=1;j<=p-1-i;++j) exp.emplace_back(i,j); if (got!=exp) fail("wedge_face_order_tri","tri lexicographic"); }
      if (sz==4){ std::vector<std::pair<int,int>> exp; for (int i=1;i<=p-1;++i) for (int j=1;j<=p-1;++j) exp.emplace_back(i,j); if (got!=exp) fail("wedge_face_order_quad","quad row‑major"); }
    }
  }
  // Pyramid p=3: base quad row‑major, sides tri lexicographic
  {
    int p=3; auto pat=CellTopology::high_order_pattern(CellFamily::Pyramid,p, CellTopology::HighOrderKind::Lagrange); auto fv=CellTopology::get_oriented_boundary_faces_view(CellFamily::Pyramid);
    for (int fi=0; fi<fv.face_count; ++fi) {
      int sz=fv.offsets[fi+1]-fv.offsets[fi]; auto got=face_roles(pat,fi);
      if (sz==3){ std::vector<std::pair<int,int>> exp; for (int i=1;i<=p-2;++i) for (int j=1;j<=p-1-i;++j) exp.emplace_back(i,j); if (got!=exp) fail("pyr_face_order_tri","tri lexicographic"); }
      if (sz==4){ std::vector<std::pair<int,int>> exp; for (int i=1;i<=p-1;++i) for (int j=1;j<=p-1;++j) exp.emplace_back(i,j); if (got!=exp) fail("pyr_face_order_quad","quad row‑major"); }
    }
  }

  if (fails==0) { std::cout<<"All topology sanity checks passed."<<"\n"; return 0; }
  std::cout<<fails<<" checks failed."<<"\n"; return 1;
}
