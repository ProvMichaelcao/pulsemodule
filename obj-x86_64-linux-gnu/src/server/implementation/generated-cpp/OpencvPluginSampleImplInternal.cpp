/* Autogenerated with kurento-module-creator */

#include <gst/gst.h>
#include "MediaPipeline.hpp"
#include "OpencvPluginSampleImpl.hpp"
#include "OpencvPluginSampleImplFactory.hpp"
#include "OpencvPluginSampleInternal.hpp"
#include <jsonrpc/JsonSerializer.hpp>
#include <KurentoException.hpp>

using kurento::KurentoException;

namespace kurento
{
namespace module
{
namespace opencvpluginsample
{

MediaObjectImpl *OpencvPluginSampleImplFactory::createObjectPointer (const boost::property_tree::ptree &conf, const Json::Value &params) const
{
  kurento::JsonSerializer s (false);
  OpencvPluginSampleConstructor constructor;

  s.JsonValue = params;
  constructor.Serialize (s);

  return createObject (conf, constructor.getMediaPipeline() );
}

void
OpencvPluginSampleImpl::invoke (std::shared_ptr<MediaObjectImpl> obj, const std::string &methodName, const Json::Value &params, Json::Value &response)
{
  if (methodName == "setFilterType") {
    kurento::JsonSerializer s (false);
    OpencvPluginSampleMethodSetFilterType method;

    s.JsonValue = params;
    method.Serialize (s);

    method.invoke (std::dynamic_pointer_cast<OpencvPluginSample> (obj) );
    return;
  }

  if (methodName == "setEdgeThreshold") {
    kurento::JsonSerializer s (false);
    OpencvPluginSampleMethodSetEdgeThreshold method;

    s.JsonValue = params;
    method.Serialize (s);

    method.invoke (std::dynamic_pointer_cast<OpencvPluginSample> (obj) );
    return;
  }

  OpenCVFilterImpl::invoke (obj, methodName, params, response);
}

bool
OpencvPluginSampleImpl::connect (const std::string &eventType, std::shared_ptr<EventHandler> handler)
{

  return OpenCVFilterImpl::connect (eventType, handler);
}

void
OpencvPluginSampleImpl::Serialize (JsonSerializer &serializer)
{
  if (serializer.IsWriter) {
    try {
      Json::Value v (getId() );

      serializer.JsonValue = v;
    } catch (std::bad_cast &e) {
    }
  } else {
    throw KurentoException (MARSHALL_ERROR,
                            "'OpencvPluginSampleImpl' cannot be deserialized as an object");
  }
}
} /* opencvpluginsample */
} /* module */
} /* kurento */

namespace kurento
{

void
Serialize (std::shared_ptr<kurento::module::opencvpluginsample::OpencvPluginSampleImpl> &object, JsonSerializer &serializer)
{
  if (serializer.IsWriter) {
    if (object) {
      object->Serialize (serializer);
    }
  } else {
    std::shared_ptr<kurento::MediaObjectImpl> aux;
    aux = kurento::module::opencvpluginsample::OpencvPluginSampleImplFactory::getObject (JsonFixes::getString(serializer.JsonValue) );
    object = std::dynamic_pointer_cast<kurento::module::opencvpluginsample::OpencvPluginSampleImpl> (aux);
  }
}

void
Serialize (std::shared_ptr<kurento::module::opencvpluginsample::OpencvPluginSample> &object, JsonSerializer &serializer)
{
  std::shared_ptr<kurento::module::opencvpluginsample::OpencvPluginSampleImpl> aux = std::dynamic_pointer_cast<kurento::module::opencvpluginsample::OpencvPluginSampleImpl> (object);

  Serialize (aux, serializer);
  object = std::dynamic_pointer_cast <kurento::module::opencvpluginsample::OpencvPluginSample> (aux);
}

} /* kurento */