#include <iostream>
#include <string>
#include <vector>
#include <hdf5/openmpi/H5Cpp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/random_sample.h>
#include <filesystem>
#include <algorithm>

#include <string>
using namespace H5;
using namespace std;
static string datapath="/media/alperen/Yerel Disk/Downloads/rgbd-dataset_pcd/rgbd-dataset/";
static hsize_t pointnumber=1024;

void writePointCloudToHDF5(const vector<string> pcdFiles, const vector<string> labels,const vector<string> instances, const string& hdf5File) {
    // Open HDF5 file
    H5File file(hdf5File, H5F_ACC_TRUNC);
    pcl::RandomSample<pcl::PointXYZ> sampler;
    sampler.setSample(pointnumber);
    const int chunk_size=1;
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);

    hsize_t labelDims[1] = {pcdFiles.size()};
    DataSpace instanceDataspace(1, labelDims);
    DataSpace labelDataspace(1, labelDims);

    float data[pointnumber][3];
    int count=0;
    hsize_t offset1=0;
    hsize_t dims[3] = {pcdFiles.size(),pointnumber, 3};
    hsize_t chunkDims[3] = {chunk_size, pointnumber, 3};
    string datasetName = "pointcloud";
    DSetCreatPropList propList,propList2;
    propList.setChunk(3, chunkDims);
    hsize_t chunkDims2[1] = {1};
    propList2.setChunk(1, chunkDims2);
    DataSpace dataspace(3, dims);
    DataSet dataset = file.createDataSet(datasetName, PredType::NATIVE_FLOAT, dataspace,propList);
    DataSet labelDataset = file.createDataSet("labels", str_type, labelDataspace,propList2);
    DataSet instanceDataset = file.createDataSet("instances", str_type, labelDataspace,propList2);
    for (size_t i = 0; i < pcdFiles.size(); ++i) {
        if(i%100==0){cout<<"Written: "<<i<<" files."<<endl;}

        // Read PCD file

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ> sampled_cloud;
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcdFiles[i], *cloud) == -1) {
            PCL_ERROR("Couldn't read file %s\n", pcdFiles[i].c_str());
            continue;
        }

        sampler.setInputCloud(cloud);
        sampler.filter(sampled_cloud);
        // Prepare data for HDF5
         // number of points and 3 coordinates (x, y, z)

        for (size_t j = 0; j < sampled_cloud.size(); ++j) {
            data[j][0] = sampled_cloud.points[j].x;
            data[j][1] = sampled_cloud.points[j].y;
            data[j][2] = sampled_cloud.points[j].z;
        }

        hsize_t offset[3] = {i, 0, 0};
        DataSpace memspace(3, chunkDims);
        dataspace.selectHyperslab(H5S_SELECT_SET, chunkDims, offset);
        dataset.write(data, PredType::NATIVE_FLOAT, memspace, dataspace);

        offset1++;
        count++;
        hsize_t offset2[1] = {i};
        DataSpace memspace2(1, chunkDims2);
        DataSpace memspace3(1, chunkDims2);
        labelDataspace.selectHyperslab(H5S_SELECT_SET, chunkDims, offset2);
        instanceDataspace.selectHyperslab(H5S_SELECT_SET, chunkDims, offset2);
        labelDataset.write(labels[i], str_type,memspace2,labelDataspace);
        instanceDataset.write(instances[i], str_type,memspace3,instanceDataspace);
    }


}

static void files2read(vector<string> &filenames, vector<string> &class_names, vector<string> &instance_names){




    for (auto const& class_name : filesystem::directory_iterator(datapath) ){
        for(auto const& instance_name: filesystem::directory_iterator(class_name.path())){
            int count=0;
            for(auto const& filename: filesystem::directory_iterator(instance_name.path())){

                //if(filename.path().string().substr(filename.path().string().size()-5,5)!="1.pcd"&&filename.path().string().substr(filename.path().string().size()-5,5)!="6.pcd"){ continue;}
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
                pcl::PointCloud<pcl::PointXYZ>::Ptr sampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename.path(), *cloud) == -1) {
                    PCL_ERROR("Couldn't read file %s\n", filename.path().stem().c_str());
                    continue;
                }
                else if(cloud->width<pointnumber){
                    continue;
                }
                filenames.push_back(filename.path());
                class_names.push_back(class_name.path().stem());
                instance_names.push_back(instance_name.path().stem());

                count++;
            }
            cout<<instance_names[instance_names.size()-1]<<endl;
        }
    }
}

int main() {



    vector<string> pcdFiles,labels,instances;
    files2read(pcdFiles,labels,instances);
    string hdf5File = "wod_full_"+to_string(pointnumber)+".h5";
    writePointCloudToHDF5(pcdFiles, labels,instances, hdf5File);

    cout << "Finished writing to " << hdf5File << endl;
    return 0;
}
