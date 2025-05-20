var currentSceneList = ["Lego", "Ficus", "Hotdog", "Armadillo"];
var currentScene = "Lego";
var currentSceneId = 0;
var currentResultList = ["normal", "albedo", "rendering"];
var currentResult = "normal";
var currentResultId = 0;
var currentMethodList = ["ours", "ours_multi_light", "nerfactor", "invRender", "GT"];
var currentMethod = "ours";



function ChangeScene(idx){
  var li_list = document.getElementById("synthetic-view-ul").children;

  // console.log(idx);
  // console.log(li_list);
  for(i = 0; i < li_list.length; i++){
      li_list[i].className = "";
  }
  li_list[idx].className = "active";

  currentScene = currentSceneList[idx];
  currentSceneId = idx;
  let video = document.getElementById("TensoIR_Synthetic_Reconstruction")
  // console.log(video);
  old_src = video.src;
  // console.log(old_src);
  new_scr = old_src.split('/');
  new_scr[new_scr.length-2] = currentScene;
  new_video_dir = new_scr.join('/');
  // console.log(new_video_dir);
  video.src = new_video_dir;


    
}

function ChangeResult(idx){
  var li_list = document.getElementById("result-view-ul").children;

  // console.log(idx);
  // console.log(li_list);
  for(i = 0; i < li_list.length; i++){
      li_list[i].className = "";
  }
  li_list[idx].className = "active";

  currentResult= currentResultList[idx];
  currentResultId = idx;
  let video = document.getElementById("TensoIR_Synthetic_Reconstruction")
  // console.log(video);
  old_src = video.src;
  // console.log(old_src);
  new_scr = old_src.split('/');
  new_scr[new_scr.length-1] = currentResult + ".mp4";
  new_video_dir = new_scr.join('/');
  // console.log(new_video_dir);
  video.src = new_video_dir;
}



function ChangeVisibility(idx) {
  var li_list = document.getElementById("visibility").children;

  // console.log(idx);
  // console.log(li_list);
  for(i = 0; i < li_list.length; i++){
      li_list[i].className = "";
  }
  li_list[idx].className = "active";

  currentScene = currentSceneList[idx];
  currentSceneId = idx;
  let video = document.getElementById("TensoIR_Synthetic_Visibility")
  // console.log(video);
  old_src = video.src;
  // console.log(old_src);
  new_scr = old_src.split('/');
  new_scr[new_scr.length-2] = currentScene;
  new_video_dir = new_scr.join('/');
  // console.log(new_video_dir);
  video.src = new_video_dir;

}

function ChangeRelighting(idx) {

  var li_list = document.getElementById("relighting").children;

  // console.log(idx);
  // console.log(li_list);
  for(i = 0; i < li_list.length; i++){
      li_list[i].className = "";
  }
  li_list[idx].className = "active";

  currentScene = currentSceneList[idx];
  currentSceneId = idx;
  let video = document.getElementById("TensoIR_Synthetic_Relighting")
  // console.log(video);
  old_src = video.src;
  // console.log(old_src);
  new_scr = old_src.split('/');
  new_scr[new_scr.length-1] = currentScene + ".mp4";
  new_video_dir = new_scr.join('/');
  // console.log(new_video_dir);
  video.src = new_video_dir;


}

function ChangeMaterialsEditing(idx) {
    var li_list = document.getElementById("materials_editing").children;


    for(i = 0; i < li_list.length; i++){
        li_list[i].className = "";
    }
    li_list[idx].className = "active";
    
    new_sub_dir = currentSceneList[idx]

    origin_video_dir = document.getElementById("materials_editing_video1").src;
    console.log(origin_video_dir);
    new_video_dir = origin_video_dir.split('/');
    new_video_dir[new_video_dir.length-2] = new_sub_dir;
    new_video_dir = new_video_dir.join('/');
    document.getElementById("materials_editing_video1").src = new_video_dir;

}