/* Autogenerated with kurento-module-creator */

#ifndef __OPENCV_PLUGIN_SAMPLE_IMPL_FACTORY_HPP__
#define __OPENCV_PLUGIN_SAMPLE_IMPL_FACTORY_HPP__

#include "OpencvPluginSampleImpl.hpp"
#include "OpenCVFilterImplFactory.hpp"
#include <Factory.hpp>
#include <MediaObjectImpl.hpp>
#include <boost/property_tree/ptree.hpp>

namespace kurento
{
namespace module
{
namespace opencvpluginsample
{

class OpencvPluginSampleImplFactory : public virtual OpenCVFilterImplFactory
{
public:
  OpencvPluginSampleImplFactory () {};

  virtual std::string getName () const {
    return "OpencvPluginSample";
  };

private:

  virtual MediaObjectImpl *createObjectPointer (const boost::property_tree::ptree &conf, const Json::Value &params) const;

  MediaObjectImpl *createObject (const boost::property_tree::ptree &conf, std::shared_ptr<MediaPipeline> mediaPipeline) const;
};

} /* opencvpluginsample */
} /* module */
} /* kurento */

#endif /*  __OPENCV_PLUGIN_SAMPLE_IMPL_FACTORY_HPP__ */